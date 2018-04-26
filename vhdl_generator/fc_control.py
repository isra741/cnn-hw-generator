import cnn_types as ct
import numpy as np
import math as mt
import os

def gen_fc_control():
    for l in range(ct.fc):
        la = l + 1
        file_name = os.path.join(ct.vhdl_path, 'fc'+ str(la)+ '_control.vhd')
        f = open(file_name, 'w')

        f.write('library IEEE;\n')
        f.write('use IEEE.STD_LOGIC_1164.ALL;\n')
        f.write('use IEEE.NUMERIC_STD.ALL;\n')
        f.write('use ieee.std_logic_UNSIGNED.ALL;\n')
        f.write('use WORK.CNN_PACKAGE.ALL;\n\n')

        f.write('entity fc{0}_control is\n'.format(la))
        f.write('    Port (clk : in STD_LOGIC;\n')
        f.write('          rst_n : in STD_LOGIC;\n')
        f.write('          fc{0}_start : in STD_LOGIC;\n'.format(la))
        if l == 0:
            f.write('          fc{0}_in: in POOL{1}_IMG_FC{2};\n'.format(la, ct.pool, 1))
        else:
            f.write('          fc{0}_in: in FC{1}_RELU_LAYER;\n'.format(la, l))
        f.write('          fc{0}_macc_out_layer: in FC{1}_MACC_OUT_LAYER;\n\n'.format(la, la))
                               
        f.write('          fc{0}_ce: out STD_LOGIC;\n'.format(la))
        f.write('          fc{0}_macc_rst: out STD_LOGIC;\n'.format(la))
        f.write('          fc{0}_carryin: out STD_LOGIC;\n'.format(la))
        f.write('          fc{0}_load : out STD_LOGIC;\n'.format(la))
        if l == (ct.fc - 1):
            f.write('          fc{0}_done : out STD_LOGIC;\n\n'.format(la))
        else:
            f.write('          fc{0}_start : out STD_LOGIC;\n\n'.format(la + 1))          
        f.write('          fc{0}_relu_layer: out FC{1}_RELU_LAYER;\n'.format(la, la))
        f.write('          fc{0}_load_data : out FC{1}_BIAS_LAYER;\n'.format(la, la))
        f.write('          fc{0}_weight_mux : out FC{1}_WEIGHT_MUX;\n'.format(la, la))
        if l == 0:
            f.write('          fc{0}_input: out CONV{1}_OUT\n'.format(la, ct.conv))    
        else:
            f.write('          fc{0}_input: out FC{1}_RELU\n'.format(la, l))
        f.write('          );\n')
        f.write('end fc{0}_control;\n\n'.format(la))

        f.write('architecture Behavioral of fc{0}_control is\n'.format(la))
        f.write('type states is (RST_FC, LOAD_DATA, READ_START, CHECK_START, MACC,\n') 
        f.write('END_INNER,  WRITE_DONE, BACK);\n')

        f.write('signal fc{0}_state: states;\n\n'.format(la))

        if l == 0:
            fn = int(mt.ceil(mt.log(ct.conv_feat[ct.conv - 1], 2)))
            pn = int(mt.ceil(mt.log(ct.p2_dim**2, 2)))
            f.write('signal feat_count: UNSIGNED({0} downto 0);\n'.format(fn))
            f.write('signal pix_count: UNSIGNED({0} downto 0);\n\n'.format(pn))
        else:
            i_n = int(mt.ceil(mt.log(ct.fc_neurons[l - 1], 2)))
            f.write('signal inner_count: UNSIGNED({0} downto 0);\n\n'.format(i_n)) 

        f.write('begin\n')

        f.write('fc{0}_control: process(clk, rst_n)\n'.format(la))

        if l == 0:
            f.write('        variable f: integer;\n')
            f.write('        variable p: integer;\n')
            f.write('        variable index: integer;\n')
        else:
            f.write('        variable i: integer;\n')
            
        f.write('    begin\n')
        f.write('        -- application reset\n')
        #Reset
        f.write("    if rst_n = '0' then\n")  
        f.write('        --DSP signals\n')
        f.write("        fc{0}_CE <='0';\n".format(la)) 
        f.write("        fc{0}_LOAD <= '0';\n".format(la)) 
        f.write("        fc{0}_macc_rst <= '1';\n".format(la)) 
        f.write("        fc{0}_CARRYIN <= '0';\n".format(la))

        if l == 0:
            f.write("        feat_count <= (others => '0');\n") 
            f.write("        pix_count <= (others => '0');\n")
        else:
            f.write("        inner_count <= (others => '0');\n")
            
        f.write("        fc{0}_input <= (others => '0');\n\n".format(la))
                
        f.write('        --weights and input signals\n')
        f.write('        for n in 0 to (fc{0}_neurons - 1) loop\n'.format(la))
        f.write("            fc{0}_load_data(n) <= (others => '0');\n".format(la))
        f.write("            fc{0}_weight_mux(n) <= (others => '0');\n".format(la))
        f.write("            fc{0}_relu_layer(n) <= (others => '0');\n".format(la)) 
        f.write('        end loop;\n\n') 

        if l == (ct.fc - 1):
            f.write("        fc{0}_done <= '0';\n".format(la))
        else:
            f.write("        fc{0}_start <= '0';\n".format(la + 1))

        f.write('        fc{0}_state <= RST_FC;\n\n'.format(la))
                              
        f.write('    -- application main\n')
        f.write('    elsif rising_edge(clk) then\n')
        f.write('        -- FSM ---\n')
        f.write('        case (fc{0}_state) is\n'.format(la))

        #Reset state
        f.write('            when RST_FC =>\n')
        f.write('                --DSP signals\n')
        f.write("                fc{0}_CE <='0';\n".format(la))
        f.write("                fc{0}_LOAD <= '0';\n".format(la)) 
        f.write("                fc{0}_macc_rst <= '1';\n".format(la)) 
        f.write("                fc{0}_CARRYIN <= '0';\n".format(la))

        if l == 0:
            f.write("                feat_count <= (others => '0');\n") 
            f.write("                pix_count <= (others => '0');\n")
        else:
            f.write("                inner_count <= (others => '0');\n")
            
        f.write("                fc{0}_input <= (others => '0');\n\n".format(la)) 
                        
        f.write('                --weights signals\n')
        f.write('                for n in 0 to (fc{0}_neurons - 1) loop\n'.format(la))
        f.write("                    fc{0}_load_data(n) <= (others => '0');\n".format(la))
        f.write("                    fc{0}_weight_mux(n) <= (others => '0');\n".format(la))
        f.write("                    fc{0}_relu_layer(n) <= (others => '0');\n".format(la)) 
        f.write('                end loop;\n\n')

        if l == (ct.fc - 1):   
            f.write("                fc{0}_done <= '0';\n".format(la))
        else:
            f.write("                fc{0}_start <= '0';\n".format(la + 1))
            
        f.write('                fc{0}_state <= LOAD_DATA;\n'.format(la))

        #State to load the bias
        f.write('            when LOAD_DATA =>\n')
        f.write("                fc{0}_macc_rst <= '0';\n".format(la)) 
        if l == 0:
            f.write("                feat_count <= (others => '0');\n") 
            f.write("                pix_count <= (others => '0');\n") 
        else:
            f.write("                inner_count <= (others => '0');\n")
            
        f.write("                fc{0}_LOAD <= '1';\n".format(la)) 
        f.write("                fc{0}_CE <= '1';\n".format(la)) 
        f.write('                for n in 0 to (fc{0}_neurons - 1) loop\n'.format(la))
        f.write("                    fc{0}_load_data(n) <= fcw{0}b(n);\n".format(la)) 
        f.write('                end loop;\n\n')
        f.write('                fc{0}_state <= READ_START;\n'.format(la))

        #State that waits for a start signal
        f.write('            when READ_START =>\n')
        f.write("                fc{0}_LOAD <= '0';\n".format(la)) 
        f.write("                fc{0}_CE <= '0';\n".format(la)) 
        f.write("                if fc{0}_start = '1' then\n".format(la)) 
        f.write('                    fc{0}_state <= MACC;\n'.format(la))
        f.write('                end if;\n')

        #MACC state to compute the inner product
        f.write('            when MACC =>\n')

        if l == 0:
            fn = int(mt.ceil(mt.log(ct.conv_feat[ct.conv - 1], 2)))
            pn = int(mt.ceil(mt.log(ct.p2_dim**2, 2)))
            f.write('                if pix_count < TO_UNSIGNED((p{0}_window - 1), {1}) then\n'.format(ct.pool, pn + 1))
            f.write('                    pix_count <= pix_count + TO_UNSIGNED(1, {0});\n'.format(pn + 1))
            f.write('                elsif pix_count = TO_UNSIGNED(p{0}_window - 1, {1})\n'.format(ct.pool, pn + 1)) 
            f.write('                and feat_count < TO_UNSIGNED(c{0}_feat_maps - 1, {1}) then\n'.format(ct.conv, fn + 1))
            f.write("                    pix_count <= (others => '0');\n") 
            f.write('                    feat_count <= feat_count + TO_UNSIGNED(1, {0});\n'.format(fn + 1))
            f.write('                elsif pix_count = TO_UNSIGNED(p{0}_window - 1, {1})\n'.format(ct.pool, pn + 1)) 
            f.write('                and feat_count = TO_UNSIGNED(c{0}_feat_maps - 1, {1}) then\n'.format(ct.conv, fn + 1))
        else:
            i_n = int(mt.ceil(mt.log(ct.fc_neurons[l - 1], 2)))
            f.write('                if inner_count < TO_UNSIGNED((fc{0}_neurons - 1), {1}) then\n'.format(la - 1, i_n + 1))
            f.write('                    inner_count <= inner_count + TO_UNSIGNED(1, {0});\n'.format(i_n + 1))
            f.write('                elsif inner_count = TO_UNSIGNED(fc{0}_neurons - 1, {1}) then\n'.format(la - 1, i_n + 1))
                
        f.write('                    fc{0}_state <= END_INNER;\n'.format(la))      
        f.write('                end if;\n\n')
        f.write('                for n in 0 to (fc{0}_neurons - 1) loop\n'.format(la))

        if l == 0:
            f.write('                    f := TO_INTEGER(feat_count);\n')
            f.write('                    p := TO_INTEGER(pix_count);\n')
            f.write('                    index := f*p{0}_window + p;\n'.format(ct.pool))
            f.write('                    fc{0}_weight_mux(n) <= fcw{1}(n)(index);\n'.format(la, la))
            f.write('                end loop;\n')
            f.write('                fc{0}_input <= fc{1}_in(p)(f);\n\n'.format(la, la))        
        else:
            f.write('                    i := TO_INTEGER(inner_count);\n')
            f.write('                    fc{0}_weight_mux(n) <= fcw{1}(n)(i);\n'.format(la, la))
            f.write('                end loop;\n')
            f.write('                fc{0}_input <= fc{1}_in(i);\n\n'.format(la, la))
                
        f.write("                FC{0}_CE <= '1';\n".format(la))

        #State to reset the MACC        
        f.write('            when END_INNER =>\n')
        f.write("                FC{0}_LOAD<= '1';\n".format(la)) 
        f.write('                for n in 0 to (fc{0}_neurons - 1) loop\n'.format(la))
        f.write('                    fc{0}_load_data(n) <= fcw{1}b(n);\n'.format(la, la))
        f.write('                end loop;\n')
        f.write('                fc{0}_state <= WRITE_DONE;\n'.format(la))

        #State to perform the RELU function and reduce the precision
        f.write('            when WRITE_DONE =>\n')
        f.write("                fc{0}_ce <= '0';\n".format(la))
        f.write("                fc{0}_load <= '0';\n".format(la))
                
        if l == (ct.fc - 1):
            f.write("                fc{0}_done <= '1';\n\n".format(la))
        else:
            f.write("                fc{0}_start <= '1';\n\n".format(la + 1))
                
        f.write('                --RELU\n')
        f.write('                for n in 0 to (fc{0}_neurons - 1) loop\n'.format(la)) 
        f.write('                    if( fc{0}_macc_out_layer(n) > 0) then\n'.format(la))
        n = 1 + ct.nfoi[l]+ ct.nfwf[l] + ct.nfwf[l]
        f.write('                        fc{0}_relu_layer(n) <= fc{1}_macc_out_layer(n)({2} downto {3});\n'.format(la, la, n - 1, ct.nfwf[l]*2)) 
        f.write('                    else\n')
                
        if l == (ct.fc - 1):
            f.write('                        if fc{0}_macc_out_layer(n)({1} downto 0) /= TO_SIGNED(0, {1}) then\n'.format(la, ct.nfwf[l]*2 - 1, ct.nfwf[l]*2))
            f.write('                            fc{0}_relu_layer(n) <= fc{1}_macc_out_layer(n)({2} downto {3}) \
    + TO_SIGNED(1, {4});\n'.format(la, la, n - 1, ct.nfwf[l]*2, ct.nfwf[l]))
            f.write('                        end if;\n'.format(la))         
        else:
            f.write("                        fc{0}_relu_layer(n) <= (others => '0');\n".format(la)) 

        f.write('                    end if;\n')
        f.write('                end loop;\n')
        f.write('                fc{0}_state <= BACK;\n'.format(la))

        #State to continue back to the "read start "state
        f.write('            when BACK =>\n')
                
        if l == (ct.fc - 1):
            f.write("                fc{0}_done <= '0';\n".format(la))
        else:
            f.write("                fc{0}_start <= '0';\n".format(la + 1))
                
        f.write("                fc{0}_macc_rst <= '0';\n".format(la)) 
        f.write('                fc{0}_state <= LOAD_DATA;\n'.format(la)) 
        f.write('            when OTHERS =>\n')
        f.write('                fc{0}_state <= RST_FC;\n'.format(la))     
        f.write('        end case;\n')
        f.write('    end if;--End of reset\n')
        f.write('end process;\n\n')

        f.write('end Behavioral;\n')
        f.close()
    return

