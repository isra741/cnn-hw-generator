import cnn_types as ct
import numpy as np

def kernel_list(kernel):
    k_list = []
    for i in range(kernel):
        k_list.append(i)
    return k_list

def gen_max_pool():
    for l in range(ct.pool):
        la = l + 1
        f = open(ct.vhdl_path + '\pool{0}.vhd'.format(la), 'w')
        f.write('library IEEE;\n')
        f.write('use IEEE.STD_LOGIC_1164.ALL;\n')
        f.write('use IEEE.NUMERIC_STD.ALL;\n')
        f.write('use WORK.CNN_PACKAGE.ALL;\n\n')

        f.write('entity pool{0} is\n'.format(la))
        f.write('    Port ( clk : in STD_LOGIC;\n')
        f.write('       rst_n : in STD_LOGIC;\n')
        f.write('       enable : in STD_LOGIC;\n')
        f.write('       a : in POOL{0}_KERNEL;\n'.format(la))
        f.write('       y : out CONV{0}_OUT);\n'.format(la))
        f.write('end pool{0};\n'.format(la))
        f.write('architecture Behavioral of pool{0} is\n'.format(la))
        f.write('signal max : CONV{0}_OUT;\n'.format(la))
        f.write('begin\n')
        f.write('    process (rst_n, clk)\n')
        f.write('    begin\n')
        f.write("    if  (rst_n = '0') then\n")
        f.write("       max  <= (others => '0');\n")          
        f.write('    else\n')  
        f.write('       if rising_edge(clk) then\n')
        f.write("           if enable = '1' then\n")
        
        for i in range(ct.pool_kernel[l]**2):
            k_list = kernel_list(ct.pool_kernel[l]**2)
            if i == 0:
                k_list.remove(i)
                f.write('               if(')
                for j in range((ct.pool_kernel[l]**2 - 1)):
                    f.write('a({0}) >= a({1})'.format(i, k_list[j]))
                    if j < (ct.pool_kernel[l]**2 - 2):
                        f.write(' and ')        
                f.write(') then \n')
                f.write('                   max <=a({0});\n'.format(i))
            else:
                k_list.remove(i)
                f.write('               elsif(')
                for j in range((ct.pool_kernel[l]**2 - 1)):
                    f.write('a({0}) >= a({1})'.format(i, k_list[j]))
                    if j < (ct.pool_kernel[l]**2 - 2):
                        f.write(' and ') 
                f.write(') then \n')
                f.write('                   max <=a({0});\n'.format(i))
        f.write('               end if; \n')        
        f.write('           else\n')
        f.write('               max <= max; \n')  
        f.write('           end if;\n')
        f.write('       end if;\n')
        f.write('    end if;\n')
        f.write('    end process;\n\n')
        f.write('y <= max;\n\n')  
        f.write('end Behavioral;\n')
        f.close()
    return
