import ida_idaapi
import ida_bytes
import ida_range
import ida_kernwin as kw
import ida_hexrays as hr
import ida_funcs
import ida_diskio
import ida_ida
import ida_graph
import ida_lines
import idautils
import idaapi
import ida_moves
import idc
from ida_pro import IDA_SDK_VERSION
import json
import os
import re
import pickle

mcode_t = {
  0x00 : "m_nop"  ,
  0x01 : "m_stx"  ,
  0x02 : "m_ldx"  ,
  0x03 : "m_ldc"  ,
  0x04 : "m_mov"  ,
  0x05 : "m_neg"  ,
  0x06 : "m_lnot" ,
  0x07 : "m_bnot" ,
  0x08 : "m_xds"  ,
  0x09 : "m_xdu"  ,
  0x0A : "m_low"  ,
  0x0B : "m_high" ,
  0x0C : "m_add"  ,
  0x0D : "m_sub"  ,     
  0x0E : "m_mul"  ,     
  0x0F : "m_udiv" ,     
  0x10 : "m_sdiv" ,     
  0x11 : "m_umod" ,     
  0x12 : "m_smod" ,     
  0x13 : "m_or"   ,     
  0x14 : "m_and"  ,     
  0x15 : "m_xor"  ,     
  0x16 : "m_shl"  ,     
  0x17 : "m_shr"  ,     
  0x18 : "m_sar"  ,     
  0x19 : "m_cfadd",
  0x1A : "m_ofadd",
  0x1B : "m_cfshl",
  0x1C : "m_cfshr",
  0x1D : "m_sets" ,
  0x1E : "m_seto" ,
  0x1F : "m_setp" ,
  0x20 : "m_setnz",
  0x21 : "m_setz" ,
  0x22 : "m_setae",
  0x23 : "m_setb" ,
  0x24 : "m_seta" ,
  0x25 : "m_setbe",
  0x26 : "m_setg" ,
  0x27 : "m_setge",
  0x28 : "m_setl" ,
  0x29 : "m_setle",     
  0x2A : "m_jcnd" ,     
  0x2B : "m_jnz"  ,       
  0x2C : "m_jz"   ,     
  0x2D : "m_jae"  ,     
  0x2E : "m_jb"   ,     
  0x2F : "m_ja"   ,     
  0x30 : "m_jbe"  ,     
  0x31 : "m_jg"   ,     
  0x32 : "m_jge"  ,    
  0x33 : "m_jl"   ,     
  0x34 : "m_jle"  ,     
  0x35 : "m_jtbl" ,     
  0x36 : "m_ijmp" ,     
  0x37 : "m_goto" ,     
  0x38 : "m_call" ,     
  0x39 : "m_icall",     
  0x3A : "m_ret"  ,
  0x3B : "m_push" ,     
  0x3C : "m_pop"  ,     
  0x3D : "m_und"  ,     
  0x3E : "m_ext"  ,     
  0x3F : "m_f2i"  ,     
  0x40 : "m_f2u"  ,     
  0x41 : "m_i2f"  ,     
  0x42 : "m_u2f"  ,     
  0x43 : "m_f2f"  ,     
  0x44 : "m_fneg" ,     
  0x45 : "m_fadd" ,
  0x46 : "m_fsub" ,
  0x47 : "m_fmul" ,
  0x48 : "m_fdiv" ,
  0x49 : "m_unk"  ,
}

filter = { "deregister_tm_clones", "register_tm_clones", "__do_global_dtors_aux", "frame_dummy", "_start", "__libc_csu_init", "__libc_csu_fini", "_init", "_dl_relocate_static_pie", "_fini", "start"}

def extract_all_function():
    micro_dict = {}
    for function in idautils.Functions():
        if idc.get_segm_name(function) in [".plt", "extern", ".init", ".fini", ".plt.sec"]:
            continue
        function_name = idaapi.get_func_name(function)
        demangled = idc.get_name(function, 0x14)
        if demangled != '':
            function_name = demangled

        if function_name in filter:
            continue
 
        micro_dict[function] = { "name": function_name }
    return micro_dict


def gen_name_dict():
    name_dict = {}
    for name in idautils.Names():
        demangled = idc.get_name(name[0], 0x14)
        if demangled != '':
            name_dict[name[0]] = demangled
        else:
            name_dict[name[0]] = name[1]
    return name_dict

def gen_func_dict():
    func_dict = {}
    for func_ea in idautils.Functions():
        func_name = idaapi.get_func_name(func_ea)
        demangled = idc.get_name(func_ea, 0x14)
        if demangled != '':
            func_dict[func_ea] = demangled
        else:
            func_dict[func_ea] = func_name
    return func_dict    

def gen_str_dict():
    str_dict = {}
    for s in list(idautils.Strings()):
        str_dict[s.ea] = str(s)
    return str_dict

def gen_insn_tokens(inst, tokens):
    if inst.opcode == 0x0a or inst.opcode == 0x0b:
        tokens.append(f"$op_{mcode_t[inst.opcode]}.{inst.d.size}$")
    else:
        tokens.append(f"$op_{mcode_t[inst.opcode]}$")
    return tokens

def analyze_calc_inst(mop, tokens, global_info):
    tokens.append("(")
    tokens = analyze_mop(mop.d.l, tokens, global_info)
    tokens = gen_insn_tokens(mop.d, tokens)
    tokens = analyze_mop(mop.d.r, tokens, global_info)
    tokens = analyze_mop(mop.d.d, tokens, global_info)
    tokens.append(")")
    return tokens

def analyze_normal_inst(mop, tokens, global_info):
    if mop.d.opcode == 0x02 or mop.d.opcode == 0x03:        
        # load inst
        tokens.append("[")
    else:
        tokens = gen_insn_tokens(mop.d, tokens)
    tokens = analyze_mop(mop.d.l, tokens, global_info)
    tokens = analyze_mop(mop.d.r, tokens, global_info)
    tokens = analyze_mop(mop.d.d, tokens, global_info)
    if mop.d.opcode == 0x02 or mop.d.opcode == 0x03:        
        # store inst
        tokens.append("]")
    return tokens

def analyze_operand_inst(mop, tokens, global_info):
    if 0x1c >= mop.d.opcode >= 0xc or 0x48 >= mop.d.opcode >= 0x44 or 0x29 >= mop.d.opcode >= 0x1E:
        # calculate inst
        tokens = analyze_calc_inst(mop, tokens, global_info)
    elif mop.d.opcode == 0x0a or mop.d.opcode == 0x0b:
        tokens.append(f"$op_{mcode_t[mop.d.opcode]}.{mop.d.d.size}$")
    else:
        tokens = analyze_normal_inst(mop, tokens, global_info)
    return tokens

def gen_mop_tokens(mop, tokens, global_info):
    if mop.t == hr.mop_r:
        # register                                           
        tokens.append(mop.dstr().split(".")[0])
    elif mop.t == hr.mop_n:                                         
        # immediate number
        tokens.append(hex(mop.nnn.value))
    elif mop.t == hr.mop_str:                                       
        # immediate string
        tokens.append(f"\"{mop.cstr}\"")
    elif mop.t == hr.mop_v:
        # global value
        if mop.g in str_dict:
            # string
            tokens.append(f"\"{str_dict[mop.g]}\"")
        elif mop.g in func_dict:
            # function
            if mop.g in name_dict:
                # function have symbol (export function, api function)
                tokens.append(name_dict[mop.g].replace(".", ""))
            else:
                # other function
                global_info["function"].add(mop.g)
                tokens.append(f"$function${mop.g}%#%")         
        elif mop.g in name_dict:
            # may be function in data
            tokens.append(name_dict[mop.g].replace(".", ""))    
        else:
            # other global variable
            global_info["global"].add(mop.g)
            tokens.append(f"${mop.size}.global${mop.g}%#%")
    elif mop.t == hr.mop_b:
        # block
        tokens.append(global_info["addr_map"][f"$loc_{mop.b}$"])
    elif mop.t == hr.mop_l:                                         
        # temp variable
        offset = mop.l.idx
        # offset of variable, may be structure
        if mop.l.off == 0:
            off = ''
        else:
            off = f"@{mop.l.off}" 
        if mop.l.var().has_user_name:
            # named variable
            tokens.append(f"${mop.size}.{mop.l.var().name}{off}")
        elif mop.l.var().is_result_var:
            # return variable
            tokens.append(f"${mop.size}.result${off}")
        elif mop.l.var().is_arg_var:
            # argument
            global_info["arg"].add(offset)  
            tokens.append(f"${mop.size}.arg${offset}%#%{off}")                 
        elif mop.l.var().location.in_stack():
            # stack varibale
            global_info["stack"].add(offset)
            tokens.append(f"${mop.size}.stack${offset}%#%{off}") 
        else:   
            # other variabel
            global_info["var"].add(offset)                                   
            tokens.append(f"${mop.size}.var${offset}%#%{off}")
            
    elif mop.t == hr.mop_h:
        # help function
        tokens.append(mop.helper)
    elif mop.t == hr.mop_c:
        # switch case
        tokens.append(f"$mcase$") 
        for i in range(mop.c.size()):
            if mop.c.values[i].size() == 0:
                tokens.append(f"$default$") 
            else:
                for j in range(mop.c.values[i].size()):
                    src = mop.c.values[i][j]
                    tokens.append(f"{src}")
            dst = mop.c.targets[i]
            tokens.append(global_info["addr_map"][f"$loc_{dst}$"])
    elif mop.t == hr.mop_fn:
        # float
        try:
            tokens.append(mop.fpc.fnum.float)
        except:
            tokens.append("float")
    else:
        return tokens    
    return tokens

def analyze_function(mop, tokens, global_info):
    tokens.append("<")
    for i in range(len(mop.f.args)):
        arg = mop.f.args[i]  
        if arg.t == hr.mop_d:   
            if arg.d: 
                tokens = analyze_operand_inst(arg, tokens, global_info)                                 
        elif arg.t == hr.mop_a:         
            tokens = analyze_mop(arg, tokens, global_info)
        else:   
            tokens = gen_mop_tokens(arg, tokens, global_info)
        if i != len(mop.f.args) - 1:    
            tokens.append(",")  
    tokens = gen_mop_tokens(mop, tokens, global_info)
    tokens.append(">") 
    return tokens

def analyze_mop(mop, tokens, global_info):
    if mop.t == hr.mop_r:
        tokens = gen_mop_tokens(mop, tokens, global_info)
    elif mop.t == hr.mop_n: 
        tokens = gen_mop_tokens(mop, tokens, global_info)
    elif mop.t == hr.mop_str:   
        tokens = gen_mop_tokens(mop, tokens, global_info)
    elif mop.t == hr.mop_d: 
        if mop.d:
            tokens = analyze_operand_inst(mop, tokens, global_info)           
    elif mop.t == hr.mop_S: 
        tokens = gen_mop_tokens(mop, tokens, global_info)
    elif mop.t == hr.mop_v: 
        tokens = gen_mop_tokens(mop, tokens, global_info)
    elif mop.t == hr.mop_b:
        tokens = gen_mop_tokens(mop, tokens, global_info)
    elif mop.t == hr.mop_f:
        tokens = analyze_function(mop, tokens, global_info)
    elif mop.t == hr.mop_l: 
        tokens = gen_mop_tokens(mop, tokens, global_info)
    elif mop.t == hr.mop_a: 
        tokens = analyze_mop(mop.a, tokens, global_info)
    elif mop.t == hr.mop_h:
        tokens = gen_mop_tokens(mop, tokens, global_info)
    elif mop.t == hr.mop_c: 
        tokens = gen_mop_tokens(mop, tokens, global_info)
    elif mop.t == hr.mop_fn:
        tokens = gen_mop_tokens(mop, tokens, global_info)
    elif mop.t == hr.mop_p: 
        tokens = gen_mop_tokens(mop.pair.lop, tokens, global_info)
        tokens = analyze_mop(mop.pair.lop, tokens, global_info)
        tokens = gen_mop_tokens(mop.pair.hop, tokens, global_info)
        tokens = analyze_mop(mop.pair.hop, tokens, global_info)
    return tokens

def analyze_inst(inst, global_info):
    tokens = []
    if inst:
        tokens = gen_insn_tokens(inst, tokens)    
        tokens = analyze_mop(inst.l, tokens, global_info)  
        tokens = analyze_mop(inst.r, tokens, global_info)  
        tokens = analyze_mop(inst.d, tokens, global_info)  
    return tokens

def normalize(tokens, global_info):
    global_info["arg"] = sorted(list(global_info["arg"]))               
    global_info["var"] = sorted(list(global_info["var"]))               
    global_info["stack"] = sorted(list(global_info["stack"]))           
    global_info["global"] = sorted(list(global_info["global"]))         
    global_info["function"] = sorted(list(global_info["function"]))     
    
    tokens = json.dumps(tokens) 
    
    # rebase
    for i in range(len(global_info["arg"])):
        tokens = tokens.replace("arg$" + str(global_info["arg"][i]) + "%#%", f"arg${i}")

    for i in range(len(global_info["var"])):
        tokens = tokens.replace("var$" + str(global_info["var"][i]) + "%#%", f"var${i}")

    for i in range(len(global_info["stack"])):
        tokens = tokens.replace("stack$" + str(global_info["stack"][i]) + "%#%", f"stack${i}")

    for i in range(len(global_info["global"])):
        tokens = tokens.replace("global$" + str(global_info["global"][i]) + "%#%", f"global${i}")

    for i in range(len(global_info["function"])):
        tokens = tokens.replace("function$" + str(global_info["function"][i]) + "%#%", f"function${i}")

    return json.loads(tokens)                                           

def parse_mcall(tokens, global_info):
    # normalize some call target
    for loc in tokens:
        for i in range(len(tokens[loc])):
            if tokens[loc][i] == "$op_m_call$":
                call_func = tokens[loc][i+1]
                if ".global$" in call_func:
                    index = tokens[loc][i+1].find('$', 2) + 1
                    address = tokens[loc][i+1][index:][:-3]
                    tokens[loc][i+1] = "$function$" + tokens[loc][i+1][index:]
                    global_info["function"].add(int(address))
    return tokens

def tokenizer(mba): 
    num = 0                                     
    tokens = {}
    global_info = {}                    
    global_info["arg"] = set()          
    global_info["var"] = set()          
    global_info["stack"] = set()        
    global_info["global"] = set()       
    global_info["function"] = set()     
    global_info["addr_map"] = {}        
    
    for bi in range(mba.qty):              
        blk = mba.get_mblock(bi)           
        minsn = blk.head                   
        global_info["addr_map"][f"$loc_{bi}$"] = f"$loc_{num}$"    
        if minsn == None:                           
            num += 1   
            continue
        while True:
            if minsn == None: 
                break
            minsn = minsn.next                          
            num += 1  

    for bi in range(mba.qty):
        blk = mba.get_mblock(bi)
        minsn = blk.head

        while True:
            if minsn == None:
                break
            tokens[f"$loc_{len(tokens) + 1}$"] = analyze_inst(minsn, global_info)      
            minsn = minsn.next
    
    tokens[f"$loc_{len(tokens) + 1}$"] = ["$ret$"]
    tokens = parse_mcall(tokens, global_info)
    tokens = normalize(tokens, global_info)
    return tokens, global_info

def get_call(func):
    # patch inline target
    instGenerator = idautils.FuncItems(func)
    for inst in instGenerator:

        if "call" in idc.GetDisasm(inst):
            mbr = hr.mba_ranges_t()
            mbr.ranges.push_back(ida_range.range_t(inst, inst + idc.get_item_size(inst)))

            hf = hr.hexrays_failure_t()
            ml = hr.mlist_t() 
            mba = hr.gen_microcode(mbr, hf, ml, hr.DECOMP_WARNINGS | hr.DECOMP_NO_CACHE, 8) 
            
            target = None
            for bi in range(mba.qty):
                blk = mba.get_mblock(bi)
                minsn = blk.head
                
                while True:
                    if minsn == None:
                        break
                    if minsn.l.t == hr.mop_v:
                        target = minsn.l.g
                    minsn = minsn.next

            if target != None and target in micro_dict:
                
                id = micro_dict[target]["name"]
                if id in inline_set:

                    raw_name = idaapi.get_func_name(target)
                    idc.set_name(target, "thisisatempsymbol")
                    asm = "call thisisatempsymbol"
                    ok, code = idautils.Assemble(inst, asm)

                    if ok:
                        orig_opcode_len = idc.get_item_size(inst)
                        new_code_len = len(code)
                        if orig_opcode_len >= new_code_len:
                            delta = orig_opcode_len - new_code_len
                            code += b'\x90' * delta
                            idaapi.patch_bytes(inst, code)

                    idc.set_name(target, raw_name)


if __name__=='__main__':

    idc.auto_wait()

    output_path = idc.ARGV[1]
    inline_path = idc.ARGV[2]

    inline_set = pickle.load(open(inline_path, "rb"))

    str_dict = gen_str_dict()               
    func_dict = gen_func_dict()             
    name_dict = gen_name_dict()
    micro_dict = extract_all_function() 

    to_save = {}

    for function in micro_dict:
        id = micro_dict[function]["name"]
        if id in inline_set:
            pfn = ida_funcs.get_func(function)
            pfn.flags |= 0x20000
            ida_funcs.update_func(pfn)
        get_call(function)

    for function in micro_dict:
        try:
            pfn = ida_funcs.get_func(function)
            mbr = hr.mba_ranges_t(pfn)
            hf = hr.hexrays_failure_t()
            ml = hr.mlist_t()
            mba = hr.gen_microcode(mbr, hf, ml, hr.DECOMP_WARNINGS | hr.DECOMP_NO_CACHE, 8) 
        except:
            continue
        
        if mba != None:
            tokens, global_info = tokenizer(mba)
            to_save[micro_dict[function]["name"]] = tokens   


    json.dump(to_save, open(output_path, "w"))
    idc.qexit(0)