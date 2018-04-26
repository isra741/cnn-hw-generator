import cnn_types as ct
import numpy as np
import math as mt

def gen_pool_control():
    for l in range(ct.pool):
        la = l + 1
        f = open(ct.vhdl_path + '\pool'+ str(la) + '_control.vhd', 'w')

        f.write('library IEEE;\n')
        f.write('use IEEE.STD_LOGIC_1164.ALL;\n')
        f.write('use IEEE.NUMERIC_STD.ALL;\n')
        f.write('use ieee.std_logic_UNSIGNED.ALL;\n')
        f.write('use WORK.CNN_PACKAGE.ALL;\n\n')

        f.write('entity pool{0}_control is\n'.format(la))
        f.write('    Port (clk : in STD_LOGIC;\n')
        f.write('      rst_n : in STD_LOGIC;\n')
        f.write('      pool{0}_pixel_layer: in POOL{1}_PIXEL_LAYER;\n'.format(la, la))
        f.write('      p{0}_start : in STD_LOGIC;\n\n'.format(la))

        if l == (ct.pool - 1):
            f.write('      fc{0}_start : out STD_LOGIC;\n'.format(1))
            f.write('      pool{0}_img_out: out POOL{1}_IMG_FC{2}\n'.format(la, la, 1))
        else:
            f.write('      c{0}_start : out STD_LOGIC;\n'.format(la + 1))
            f.write('      pool{0}_kernel_out: out POOL{1}_KERNEL_CONV{2}\n'.format(la, la, la + 1))
        f.write('      );\n')
        f.write('end pool{0}_control;\n\n'.format(la))

        f.write('architecture Behavioral of pool{0}_control is\n'.format(la))

        #States signlas declaration
        f.write('type states is (RST_POOL, READ_START, WRITE, DONE);\n')
        f.write('signal pool{0}_state: states;\n'.format(la))

        #Temp signals and counters
        if l == (ct.pool - 1):
            f.write('signal pool{0}_img_tmp: POOL{1}_IMG_FC{2};\n'.format(la, la, 1))
            n = int(mt.ceil(mt.log(ct.p2_dim**2, 2)))
            f.write('signal pix_count: UNSIGNED({0} downto 0);\n\n'.format(n))
        else:
            f.write('signal pool{0}_kernel_tmp: POOL{1}_KERNEL_CONV{2};\n'.format(la, la, la + 1))
            n = int(mt.ceil(mt.log(ct.conv_kernel[la]**2, 2)))
            f.write('signal conv{0}_count: UNSIGNED({1} downto 0);\n\n'.format(la + 1, n))   
        f.write('begin\n\n')

        f.write('pool{0}_control: process(clk, rst_n)\n'.format(la))
        if l == (ct.pool - 1):
            f.write('    variable p : integer :=0;\n')
        else:
            f.write('    variable c : integer :=0;\n')
        f.write('    begin\n')
        f.write('     -- application reset\n')

        #Reset
        f.write("    if rst_n = '0' then\n")
        f.write('        pool{0}_state <= RST_POOL;\n'.format(la))
       
        if l == (ct.pool - 1):
            f.write("        pix_count <= (others => '0');\n")
            f.write("        fc{0}_start <= '0';\n\n".format(1))
            f.write('        for i in 0 to (p{0}_window - 1) loop\n'.format(la))
            f.write('            for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
            f.write("                pool{0}_img_out(i)(n) <= (others => '0');\n".format(la))
            f.write("                pool{0}_img_tmp(i)(n) <= (others => '0');\n".format(la)) 
        else:
            f.write("        conv{0}_count <= (others => '0');\n".format(la + 1))
            f.write("        c{0}_start <= '0';\n\n".format(la + 1))
            f.write('        for i in 0 to (c{0}_kernel*c{1}_kernel - 1) loop\n'.format(la, la))
            f.write('            for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
            f.write("                pool{0}_kernel_out(i)(n) <= (others => '0');\n".format(la))
            f.write("                pool{0}_kernel_tmp(i)(n) <= (others => '0');\n".format(la)) 
        f.write('            end loop;\n')
        f.write('        end loop;\n\n')
                              
        f.write('    -- application main\n')
        f.write('    elsif rising_edge(clk) then\n')
        f.write('        -- FSM ---\n')
        f.write('        case (pool{0}_state) is\n'.format(la))

        #Reset state
        f.write('            when RST_POOL =>\n')
        
        
        if l == (ct.pool - 1):
            f.write("                pix_count <= (others => '0');\n")
            f.write("                fc{0}_start <= '0';\n\n".format(1))
            f.write('                for i in 0 to (p{0}_window - 1) loop\n'.format(la))
            f.write('                    for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
            f.write("                        pool{0}_img_out(i)(n) <= (others => '0');\n".format(la))
            f.write("                        pool{0}_img_tmp(i)(n) <= (others => '0');\n".format(la)) 
        else:
            f.write("                conv{0}_count <= (others => '0');\n".format(la + 1))
            f.write("                c{0}_start <= '0';\n\n".format(la + 1))
            f.write('                for i in 0 to (c{0}_kernel*c{1}_kernel - 1) loop\n'.format(la, la))
            f.write('                    for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
            f.write("                        pool{0}_kernel_out(i)(n) <= (others => '0');\n".format(la))
            f.write("                        pool{0}_kernel_tmp(i)(n) <= (others => '0');\n".format(la)) 
        f.write('                    end loop;\n')
        f.write('                end loop;\n\n')

        
        f.write('                pool{0}_state <= READ_START;\n'.format(la))

        #State to wait for a start signals
        f.write('            when READ_START =>\n')
        if l == (ct.pool - 1):
            f.write("                fc{0}_start <= '0';\n\n".format(1))
        else:
            f.write("                c{0}_start <= '0';\n\n".format(la + 1))
                        
        f.write("                if p{0}_start = '1' then\n".format(la))
        f.write('                    pool{0}_state <= WRITE;\n'.format(la))
        f.write('                end if;\n')

        #State to write the kernel of the next layer
        f.write('            when WRITE =>\n')
        if l == (ct.pool - 1):
            f.write('                p := TO_INTEGER(pix_count);\n\n')
            f.write('                for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
            f.write('                    pool{0}_img_tmp(p)(n) <= pool{1}_pixel_layer(n);\n'.format(la, la))
        else:
            f.write('                c := TO_INTEGER(conv{0}_count);\n\n'.format(la + 1))
            f.write('                for n in 0 to (c{0}_feat_maps - 1) loop\n'.format(la))
            f.write('                    pool{0}_kernel_tmp(c)(n) <= pool{1}_pixel_layer(n);\n'.format(la, la))
        
        f.write('                end loop;\n\n')
        if l == (ct.pool - 1):
            n = int(mt.ceil(mt.log(ct.p2_dim**2, 2)))
            f.write('                if pix_count < TO_UNSIGNED((p{0}_window - 1), {1}) then\n'.format(la, n + 1))
            f.write('                    pix_count <= pix_count + TO_UNSIGNED(1, {0});\n'.format(n + 1))
        else:
            n = int(mt.ceil(mt.log(ct.conv_kernel[la]**2, 2)))
            f.write('                if conv{0}_count < TO_UNSIGNED((c{1}_kernel*c{2}_kernel - 1), {3}) then\n'.format(la + 1, la + 1, la + 1, n + 1))
            f.write('                    conv{0}_count <= conv{1}_count + TO_UNSIGNED(1, {2});\n'.format(la + 1, la + 1, n + 1))
        f.write('                    pool{0}_state <= READ_START;\n'.format(la))
        f.write('                else\n')
        f.write('                    pool{0}_state <= DONE;\n'.format(la))   
        f.write('                end if;\n')

        #State to start the next layer
        f.write('            when DONE  =>\n')
        if l == (ct.pool - 1):
            f.write("                fc{0}_start <= '1';\n".format(1))
            f.write('                pool{0}_img_out <= pool{1}_img_tmp;\n'.format(la, la))
            f.write("                pix_count <= (others => '0');\n")
        else:
            f.write("                c{0}_start <= '1';\n".format(la + 1))
            f.write('                pool{0}_kernel_out <= pool{1}_kernel_tmp;\n'.format(la, la))
            f.write("                conv{0}_count <= (others => '0');\n".format(la + 1))
        
        f.write('                pool{0}_state <= READ_START;\n'.format(la)) 
        f.write('            when OTHERS  =>\n')
        f.write('                pool{0}_state <= RST_POOL;\n'.format(la))
        f.write('        end case;\n')
        f.write('    end if;--End of reset\n')
        f.write('end process;\n\n')

        f.write('end Behavioral;\n')
        f.close()
    return
