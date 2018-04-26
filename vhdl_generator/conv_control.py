import cnn_types as ct
import numpy as np
import math as mt

def gen_conv_control():
    for l in range(1, ct.conv):
        la = l + 1
        f = open(ct.vhdl_path + '\conv'+ str(la) + '_control.vhd', 'w')
        f.write('library IEEE;\n')
        f.write('use IEEE.STD_LOGIC_1164.ALL;\n')
        f.write('use IEEE.NUMERIC_STD.ALL;\n')
        f.write('use ieee.std_logic_UNSIGNED.ALL;\n')
        f.write('use WORK.CNN_PACKAGE.ALL;\n\n')

        f.write('entity conv{0}_control is\n'.format(la))
        f.write('    Port (clk : in STD_LOGIC;\n')
        f.write('          rst_n : in STD_LOGIC;\n')
        f.write('          conv{0}_macc_out_layer: in CONV{1}_MACC_OUT_LAYER;\n\n'.format(la, la))
                  
        f.write('          c{0}_start : in STD_LOGIC;\n'.format(la)) 
        f.write('          c{0}_in: in POOL{1}_KERNEL_CONV{2};\n'.format(la, l, la))
        f.write('          c{0}_input: out CONV{1}_OUT;\n\n'.format(la, l))
                    
        f.write('          c{0}_ce: out STD_LOGIC;\n'.format(la))
        f.write('          c{0}_macc_rst: out STD_LOGIC;\n'.format(la))
        f.write('          c{0}_carryin: out STD_LOGIC;\n'.format(la))
        f.write('          c{0}_load : out STD_LOGIC;\n'.format(la))  
        f.write('          p{0}_start : out STD_LOGIC;\n\n'.format(la))
        
        f.write('          conv{0}_load_data : out CONV{1}_BIAS_LAYER;\n'.format(la, la))          
        f.write('          conv{0}_weight_mux : out CONV{1}_WEIGHT_MUX;\n'.format(la, la))
        f.write('          pool{0}_kernel_layer : out POOL{1}_KERNEL_LAYER\n'.format(la, la))
        f.write('          );\n')
        f.write('end conv{0}_control;\n\n'.format(la))

        f.write('architecture Behavioral of conv{0}_control is\n'.format(la))

        #State signals and counters        
        f.write('type states is (RST_CONV, LOAD_DATA, READ_START, MACC_RST,\n') 
        f.write('CHECK_POOL, WRITE_KERNEL, MACC, END_POOL);\n')
        f.write('signal conv{0}_state: states;\n\n'.format(la))
        f.write('signal pool_in: POOL{0}_PIXEL_LAYER;\n'.format(la))
        fn = int(mt.ceil(mt.log(ct.conv_feat[l - 1], 2)))
        f.write('signal feat_count: UNSIGNED({0} downto 0);\n'.format(fn))
        kn = int(mt.ceil(mt.log(ct.conv_kernel[l]**2, 2)))
        f.write('signal conv_count: UNSIGNED({0} downto 0);\n'.format(kn))
        pn = int(mt.ceil(mt.log(ct.pool_kernel[l]**2, 2)))
        f.write('signal pool_count: UNSIGNED({0} downto 0);\n\n'.format(pn))

        f.write('begin\n\n')

        f.write('conv{0}_control: process(clk, rst_n)\n'.format(la))
        f.write('        variable f: integer;\n')
        f.write('        variable p: integer;\n')
        f.write('        variable c: integer;\n')
        f.write('    begin\n')
        f.write('        -- application reset\n')

        #Reset
        f.write("    if rst_n = '0' then\n")  
        f.write('        --DSP signals\n')
        f.write("        C{0}_CE <='0';\n".format(la))
        f.write("        C{0}_LOAD <= '0';\n".format(la))
        f.write("        c{0}_macc_rst <= '1';\n".format(la))
        f.write("        C{0}_CARRYIN <= '0';\n".format(la))
        f.write("        feat_count <= (others => '0');\n")
        f.write("        pool_count <= (others => '0');\n")
        f.write("        conv_count <= (others => '0');\n")
        f.write("        c{0}_input <= (others => '0');\n\n".format(la))
                
        f.write('        --weights and input signals\n')
        f.write('        for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
        f.write("            conv{0}_load_data(n) <= (others => '0');\n".format(la))
        f.write("            conv{0}_weight_mux(n) <= (others => '0');\n".format(la))
        f.write("            pool_in(n) <= (others => '0');\n") 
        f.write('            for p in 0 to (p{0}_kernel*p{1}_kernel - 1) loop\n'.format(la, la, la))
        f.write("                pool{0}_kernel_layer(n)(p) <= (others => '0');\n".format(la))    
        f.write('             end loop;\n')
        f.write('         end loop;\n \n')
                
        f.write("        p{0}_start <= '0';\n".format(la)) 
        f.write('         conv{0}_state <= RST_CONV;\n\n'.format(la))
                              
        f.write('     -- application main\n')
        f.write('     elsif rising_edge(clk) then\n')
        f.write('         -- FSM ---\n')
        f.write('         case (conv{0}_state) is\n'.format(la))

        #Reset state
        f.write('             when RST_CONV =>\n')
        f.write('                 --DSP signals\n')
        f.write("                C{0}_CE <='0';\n".format(la))
        f.write("                C{0}_LOAD <= '0';\n".format(la))
        f.write("                c{0}_macc_rst <= '1';\n".format(la))
        f.write("                C{0}_CARRYIN <= '0';\n".format(la))
        f.write("                feat_count <= (others => '0');\n")
        f.write("                pool_count <= (others => '0');\n")
        f.write("                conv_count <= (others => '0');\n")
        f.write("                c{0}_input <= (others => '0');\n\n".format(la))
                        
        f.write('                 --weights signals\n')
        f.write('                 for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
        f.write("                    conv{0}_load_data(n) <= (others => '0');\n".format(la))
        f.write("                    conv{0}_weight_mux(n) <= (others => '0');\n".format(la))
        f.write("                    pool_in(n) <= (others => '0');\n") 
        f.write('                     for p in 0 to (p{0}_kernel*p{1}_kernel - 1) loop\n'.format(la, la))
        f.write("                        pool{0}_kernel_layer(n)(p) <= (others => '0');\n".format(la))    
        f.write('                     end loop;\n')
        f.write('                 end loop;\n\n')
                                        
        f.write('                 conv{0}_state <= LOAD_DATA;\n'.format(la))

        #State to load the bias
        f.write('             when LOAD_DATA =>\n')
        f.write("                feat_count <= (others => '0');\n")
        f.write("                conv_count <= (others => '0');\n")
        f.write("                C{0}_LOAD <= '1';\n".format(la))
        f.write("                c{0}_macc_rst <= '0';\n".format(la))            
        f.write("                C{0}_CE <= '1';\n".format(la))
        f.write('                 for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
        f.write("                     conv{0}_load_data(n) <= cw{0}b(n);\n".format(la))
        f.write('                 end loop;\n\n')
        f.write('                 conv{0}_state <= READ_START;\n'.format(la))
        f.write('             when READ_START =>\n')
        f.write("                C{0}_LOAD <= '0';\n".format(la))
        f.write("                C{0}_CE <= '0';\n".format(la))
        f.write("                if c{0}_start = '1' then\n".format(la))
        f.write('                     conv{0}_state <= MACC;\n'.format(la))
        f.write('                 end if;\n')

        #State to perform the MACC computation
        f.write('             when MACC =>\n')
        ck = ct.conv_kernel[l]**2
        f.write('                 if conv_count < TO_UNSIGNED({0}, {1}) then\n'.format(ck - 1, kn + 1))
        f.write('                     conv_count <= conv_count + TO_UNSIGNED(1, {0});\n'.format(kn + 1))
        f.write('                 elsif conv_count = TO_UNSIGNED({0}, {1}) and feat_count < TO_UNSIGNED({2}, {3})\
    then\n'.format(ck - 1, kn + 1, ct.conv_feat[l - 1] - 1, fn + 1))
        f.write("                     conv_count <= (others => '0');\n")
        f.write('                     feat_count <= feat_count + TO_UNSIGNED(1, {0});\n'.format(fn + 1))
        f.write('                 elsif conv_count = TO_UNSIGNED({0}, {1}) and feat_count = TO_UNSIGNED({2}, {3})\
    then\n'.format(ck - 1, kn + 1, ct.conv_feat[l - 1] - 1, fn + 1))
        f.write('                     conv{0}_state <= MACC_RST;\n'.format(la))      
        f.write('                end if;\n')
        f.write('                for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
        f.write('                    f := TO_INTEGER(feat_count);\n')
        f.write('                    c := TO_INTEGER(conv_count);\n')
        f.write('                    conv{0}_weight_mux(n) <= cw{1}(n)(f)(c);\n'.format(la, la))
        f.write('                end loop;\n')
        f.write('                c{0}_input <= c{1}_in(c)(f);\n'.format(la, la))
        f.write("                C{0}_CE <= '1';\n".format(la))

        #State to rst the MACC
        f.write('            when MACC_RST =>\n')
        f.write("                C{0}_LOAD <= '1';\n".format(la))
        f.write('                for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
        f.write("                    conv{0}_load_data(n) <= (others => '0');\n".format(la))
        f.write('                end loop;\n\n')
        f.write('                conv{0}_state <= WRITE_KERNEL;\n'.format(la))

        #State to change the precisional and fill the pool kernel
        f.write('            when WRITE_KERNEL =>\n')
        f.write("                C{0}_LOAD <= '0';\n".format(la))
        f.write("                c{0}_CE <= '0';\n".format(la))
        f.write('                conv{0}_state <= CHECK_POOL;\n\n'.format(la))
                        
        f.write('                --float to int\n')
        f.write('                for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
        f.write('                    if( conv{0}_macc_out_layer(n) >= 0) then\n'.format(la))
        n = 1 + ct.ncoi[l] + ct.ncwf[l] + ct.ncwf[l]
        f.write('                        pool_in(n) <= conv{0}_macc_out_layer(n)({1} downto {2});\n'.format(la, n - 1, ct.ncwf[l]*2))
        f.write('                    else\n')
        f.write('                        if conv{0}_macc_out_layer(n)({1} downto 0) /= TO_SIGNED(0, {1}) then\n'.format(la, ct.ncwf[l]*2 - 1, ct.ncwf[l]*2))
        f.write('                            pool_in(n) <=conv{0}_macc_out_layer(n)({1} downto {2}) \
    + TO_SIGNED(1, {3});\n'.format(la, n - 1, ct.ncwf[l]*2, ct.ncoi[l] + 1))
        f.write('                        end if;\n')
        f.write('                    end if;\n')
        f.write('                end loop;\n')

        #Pool kernel checking
        f.write('            when CHECK_POOL =>\n')
        f.write('                pool_count <= pool_count + TO_UNSIGNED(1, {0});\n\n'.format(pn + 1))
                        
        f.write('                if pool_count < TO_UNSIGNED(p{0}_kernel*p{1}_kernel, {2}) then\n'.format(la, la, pn + 1))
        f.write('                    p := TO_INTEGER(pool_count);\n\n')
                            
        f.write('                     for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
        f.write('                         pool{0}_kernel_layer(n)(p) <= pool_in(n);\n'.format(la))
        f.write('                     end loop;\n')
        f.write('                     conv{0}_state <= LOAD_DATA;\n'.format(la))
        pk = ct.pool_kernel[l]**2
        f.write('                     if pool_count = TO_UNSIGNED({0}, {1}) then\n'.format(pk - 1, pn + 1))
        f.write('                         conv{0}_state <= END_POOL;\n'.format(la))
        f.write("                        p{0}_start <= '1';\n".format(la)) 
        f.write('                     end if;\n') 
        f.write('                 end if;\n')

        #Pool layer start
        f.write('             when END_POOL =>\n')
        f.write("                    p{0}_start <= '0';\n".format(la))
        f.write("                    pool_count <= (others => '0');\n")
        f.write('                    conv{0}_state <= LOAD_DATA;\n'.format(la))    
        f.write('             when OTHERS =>\n')
        f.write('                 conv{0}_state <= RST_CONV;\n'.format(la))
        f.write('         end case;\n\n')
                
        f.write('     end if;--End of reset\n')
        f.write(' end process;\n\n')

        f.write(' end Behavioral;\n\n')
        f.close()
    return
