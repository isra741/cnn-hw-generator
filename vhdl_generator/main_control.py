import cnn_types as ct
import numpy as np
import math as mt

def gen_main_control():
    dim = ['x', 'y']

    f = open(ct.vhdl_path + '\conv1_control.vhd', 'w')
    f.write('library IEEE;\n')
    f.write('use IEEE.STD_LOGIC_1164.ALL;\n')
    f.write('use IEEE.NUMERIC_STD.ALL;\n')
    f.write('use ieee.std_logic_UNSIGNED.ALL;\n')
    f.write('use WORK.CNN_PACKAGE.ALL;\n\n')

    f.write('entity conv1_control is\n')
    f.write('    Port (clk : in STD_LOGIC;\n')
    f.write('          rst_n : in STD_LOGIC;\n')  
    f.write('          conv1_macc_out_layer: in CONV1_MACC_OUT_LAYER;\n\n')
              
    f.write('          fc{0}_done: in STD_LOGIC;\n'.format(ct.fc))
    f.write('          fc{0}_relu_layer: in FC{0}_RELU_LAYER;\n'.format(ct.fc))
    f.write('          mem_dout : in STD_LOGIC_VECTOR(31 downto 0);\n')                           
    f.write('          mem_addr: out STD_LOGIC_VECTOR(31 downto 0);\n')
    f.write('          mem_en: out STD_LOGIC;\n')
    f.write('          mem_din : out STD_LOGIC_VECTOR(31 downto 0);\n')
    f.write('          mem_we : out STD_LOGIC_VECTOR(3 downto 0);\n')
              
    f.write('          c1_ce: out STD_LOGIC;\n')
    f.write('          c1_macc_rst: out STD_LOGIC;\n')
    f.write('          c1_carryin: out STD_LOGIC;\n')
    f.write('          c1_load : out STD_LOGIC;\n')
    f.write('          p1_start : out STD_LOGIC;\n\n')
    
    f.write('          conv1_load_data: out CONV1_BIAS_LAYER;\n')                        
    f.write('          conv1_weight_mux : out CONV1_WEIGHT_MUX;\n')
    f.write('          pool1_kernel_layer: out POOL1_KERNEL_LAYER\n')
    f.write('          );\n')
    f.write('end conv1_control;\n\n')

    f.write('architecture Behavioral of conv1_control is\n')
    f.write('type states is (RST_CONV, INIT, READ_START, OFF_RST, MACC,\n')
    f.write('LAST_MACC, MACC_RST, END_CONV1,\n')

    #State for checking the offsets
    for l in range(len(ct.feat_layers)):
        f.write('CHECK_{0}, '.format(ct.feat_layers[l].upper()))
        if l == len(ct.feat_layers) - 1:
            f.write('\n')   
    f.write('READ_FC{0}_DONE, WRITE_OUT, WRITE_DONE, DELAY, BACK);\n\n'.format(ct.fc))

    #Offset counters
    for l in range(len(ct.feat_layers)):
        for d in range(len(dim)):
            f.write('signal {0}_offset_{1} : UNSIGNED({2} downto 0);\n'.format(ct.feat_layers[l], dim[d], ct.n_offset[l][d]))
            
    c1n = int(mt.ceil(mt.log(ct.conv_kernel[0], 2)))

    for d in range(len(dim)):
        f.write('signal conv1_count_{0} : UNSIGNED({1} downto 0);\n'.format(dim[d], c1n))

    fn = int(mt.ceil(mt.log(ct.fc_neurons[ct.fc - 1], 2)))
    f.write('signal out_count : UNSIGNED({0} downto 0);\n\n'.format(fn))

    f.write('signal pool_in: POOL1_PIXEL_LAYER;\n')
    f.write('signal conv1_state: states;\n\n')

    f.write('begin\n\n')

    #Reset
    f.write('conv1_control: process(clk, rst_n)\n')
    f.write('        variable i: integer;\n') 
    f.write('        variable j: integer;\n')
    f.write('        variable o: integer;\n')
    f.write('        variable index: integer;\n')
    f.write('        variable prev_index : integer;\n')
    f.write('        variable mem_index: integer;\n')
    f.write('        variable pool_index : integer;\n')
    f.write('        variable pool_x: integer;\n')
    f.write('        variable pool_y: integer;\n')
    f.write('    begin\n')
    f.write('        -- application reset\n')
    f.write("    if rst_n = '0' then\n\n")
        
    f.write('        --Memory signals\n')
    f.write("        mem_en <= '0';\n")
    f.write("        mem_we <= (others => '0');\n")
    f.write("        mem_addr <= (others => '0');\n")
    f.write("        mem_din  <= (others => '0');\n\n")
           
    f.write('        --MACC signals\n')
    f.write("        C1_CE <='0';\n")
    f.write("        C1_LOAD <= '0';\n")
    f.write("        c1_macc_rst <= '1';\n");
    f.write("        C1_CARRYIN <= '0';\n\n")
            
    f.write('        --weights signals\n')
    f.write('        for n in 0 to (c1_feat_maps - 1) loop\n')
    f.write("            conv1_load_data(n) <= (others => '0');\n")                    
    f.write("            conv1_weight_mux(n) <= (others => '0');\n")
    f.write("            pool_in(n) <= (others => '0');\n")
    f.write('            for p in 0 to (p1_kernel*p1_kernel - 1) loop\n')
    f.write("                pool1_kernel_layer(n)(p) <= (others => '0');\n")    
    f.write('            end loop;\n')
    f.write('        end loop;\n\n')

    f.write('        --Conv offsets\n')
    for l in range(ct.conv):
        la = l + 1
        for d in range(len(dim)):
            f.write("        conv{0}_offset_{1} <= (others => '0');\n".format(la, dim[d]))
    f.write('\n        --Pool offsets\n')
    for l in range(ct.pool):
        la = l + 1
        for d in range(len(dim)):
            f.write("        pool{0}_offset_{1} <= (others => '0');\n".format(la, dim[d])) 
                   
    f.write("\n        p1_start <= '0';\n")
    f.write("        out_count <= (others => '0');\n") 
    f.write('        conv1_state <= RST_CONV;\n\n')       
                          
    f.write('    -- application main\n')
    f.write('    elsif rising_edge(clk) then\n')
    f.write('		-- FSM ---\n')
    f.write('        case conv1_state is\n')

    #Reset state
    f.write('            when RST_CONV =>\n')
    f.write('                --Memory signals\n')
    f.write("                mem_en <= '0';\n")
    f.write("                mem_we <= (others => '0');\n")
    f.write("                mem_addr <= (others => '0');\n")
    f.write("                mem_din  <= (others => '0');\n\n")
                
    f.write('                --MACC signals\n')
    f.write("                C1_CE <='0';\n")
    f.write("                C1_LOAD <= '0';\n")
    f.write("                c1_macc_rst <= '1';\n")
    f.write("                C1_CARRYIN <= '0';\n\n")
                    
    f.write('                --weights signals\n')
    f.write('                for n in 0 to (c1_feat_maps - 1) loop\n')
    f.write("                    conv1_load_data(n) <= (others => '0');\n") 
    f.write("                    conv1_weight_mux(n) <= (others => '0');\n")
    f.write("                    pool_in(n) <= (others => '0');\n") 
    f.write('                    for p in 0 to (p1_kernel*p1_kernel - 1) loop\n')
    f.write("                        pool1_kernel_layer(n)(p) <= (others => '0');\n")    
    f.write('                    end loop;\n')
    f.write('                end loop;\n\n')
                    
    f.write('                --Conv offsets\n')
    for l in range(ct.conv):
        la = l + 1
        for d in range(len(dim)):
            f.write("                conv{0}_offset_{1} <= (others => '0');\n".format(la, dim[d]))
    f.write('\n                --Pool offsets\n')
    for l in range(ct.pool):
        la = l + 1
        for d in range(len(dim)):
            f.write("                pool{0}_offset_{1} <= (others => '0');\n".format(la, dim[d]))
                    
    f.write("\n                out_count <= (others => '0');\n") 
    f.write('                conv1_state <= INIT;\n')

    #State to put the memory in read mode
    f.write('            when INIT =>\n')
    f.write('                conv1_state <= READ_START;\n')
    f.write("                c1_macc_rst <= '0';\n");
    f.write("                mem_we <= (others => '0');\n")
    f.write("                mem_addr <= (others => '0');\n")
    f.write("                mem_en <= '1';\n")

    #State to read the start signal
    f.write('            when READ_START =>\n')
    f.write('                if mem_dout = x"AAAAAAAA" then\n')
    f.write('                    conv1_state <= OFF_RST;\n')
    f.write('                end if;\n')

    #State to clear the offsets
    f.write('            when OFF_RST =>\n')
    f.write('                conv1_state <= MACC;\n')
    f.write("                conv1_count_x <= (others => '0');\n") 
    f.write("                conv1_count_y <= (others => '0');\n")

    #State to perform the macc computation of the first layer
    f.write('            when MACC =>\n')

    f.write('                if conv1_count_y < TO_UNSIGNED((c1_kernel - 1), {0})  then\n'.format(c1n + 1))
    f.write('                    conv1_count_y <= conv1_count_y + TO_UNSIGNED(1, {0});\n'.format(c1n + 1))
    f.write('                elsif conv1_count_y = TO_UNSIGNED((c1_kernel - 1), {0})\n'.format(c1n + 1))

    f.write('                and conv1_count_x < TO_UNSIGNED((c1_kernel - 1), {0}) then\n'.format(c1n + 1))
    f.write("                    conv1_count_y <= (others => '0');\n")
    f.write('                    conv1_count_x <= conv1_count_x + TO_UNSIGNED(1, {0});\n'.format(c1n + 1))
    f.write('                elsif conv1_count_y = TO_UNSIGNED((c1_kernel - 1), {0})\n'.format(c1n + 1)) 
    f.write('                and conv1_count_x = TO_UNSIGNED((c1_kernel - 1), {0}) then\n'.format(c1n + 1))

    f.write('                    conv1_state <= LAST_MACC;\n')
    f.write('                end if;\n') 
    f.write('                i := TO_INTEGER(conv1_count_x);\n')
    f.write('                j := TO_INTEGER(conv1_count_y);\n\n')
                    
    f.write('                index := j + i*c1_kernel;\n')
    f.write('                prev_index := index - 1;\n\n')
                    
    f.write('                mem_index := (j + i*width)*nbytes + start_bytes*nbytes\n')
    for l in range(len(ct.feat_layers)):
        if l == len(ct.feat_layers) - 1:
            f.write('                + TO_INTEGER({0}_offset_x) + TO_INTEGER({0}_offset_y);\n\n'.format(ct.feat_layers[l], ct.feat_layers[l]))
        else:
            f.write('                + TO_INTEGER({0}_offset_x) + TO_INTEGER({0}_offset_y)\n'.format(ct.feat_layers[l], ct.feat_layers[l]))
                    
    f.write('                mem_addr <= STD_LOGIC_VECTOR(TO_UNSIGNED(mem_index, nbytes*8));\n\n')
                    
    f.write('                if index > 0 then\n')
    f.write('                    for n in 0 to (c1_feat_maps - 1) loop\n')
    f.write('                        conv1_weight_mux(n) <= cw1(n)(prev_index);\n')
    f.write('                    end loop;\n')
    f.write('                    if index = 1 then\n')
    f.write("                        C1_LOAD <='0';\n")
    f.write('                    end if;\n')
    f.write('                else\n')
    f.write('                    for n in 0 to (c1_feat_maps - 1) loop\n')
    f.write("                        conv1_load_data(n) <= cw1b(n);\n")
    f.write('                    end loop;\n')
    f.write("                    C1_LOAD <= '1';\n")
    f.write("                    C1_CE <= '1';\n")
    f.write('                end if;\n')

    #State to peform the last macc computation
    f.write('            when LAST_MACC =>\n')
    f.write('                for n in 0 to (c1_feat_maps - 1) loop\n')
    f.write('                    conv1_weight_mux(n) <= cw1(n)({0});\n'.format(ct.conv_kernel[0]**2 - 1))
    f.write('                end loop;\n')
    f.write('                conv1_state <= MACC_RST;\n')

    #State to reset the MACC
    f.write('            when MACC_RST =>\n')
    f.write("                C1_LOAD <= '1';\n")
    f.write('                for n in 0 to (c1_feat_maps - 1) loop\n')
    f.write("                    conv1_load_data(n) <= (others => '0');\n")
    f.write('                end loop;\n')
    f.write('                conv1_state <= END_CONV1;\n')

    #State to change the precision
    f.write('            when END_CONV1 =>\n')
    f.write("                C1_LOAD <= '0';\n")
    f.write("                C1_CE <= '0';\n")
    f.write('                conv1_state <= CHECK_CONV1;\n\n')
                    
    f.write('                --float to int\n')
    f.write('                for n in 0 to (c1_feat_maps - 1) loop\n')
    f.write('                    if( conv1_macc_out_layer(n) >= 0) then\n')
    n = 1 + ct.ncoi[0]+ ct.ncwf[0] + ct.ncwf[0]
    f.write('                        pool_in(n) <= conv1_macc_out_layer(n)({0} downto {1});\n'.format(n - 1, ct.ncwf[0]*2))
    f.write('                    else\n')
    f.write('                        if conv1_macc_out_layer(n)({0} downto 0) /= TO_SIGNED(0, {1} ) then\n'.format(ct.ncwf[0]*2 - 1, ct.ncwf[0]*2))
    f.write('                            pool_in(n) <=conv1_macc_out_layer(n)({0} downto {1}) + TO_SIGNED(1, {2});\n'.format(n - 1, ct.ncwf[0]*2, ct.ncoi[0]))
    f.write('                        end if;\n')
    f.write('                    end if;\n')
    f.write('                end loop;\n')
        
    #States to check the offsets
    for l in range(len(ct.feat_layers)):
        f.write('            when CHECK_{0} =>\n'.format(ct.feat_layers[l].upper()))
        if ct.feat_layers[l] == 'conv1':
            f.write('                for n in 0 to (c1_feat_maps - 1) loop\n')
            f.write('                    pool_x := TO_INTEGER(conv1_offset_x);\n')           
            f.write('                    pool_y := TO_INTEGER(conv1_offset_y);\n')
            f.write('                    pool_index :=  p1_kernel*(pool_x/c1_step_x) + pool_y/c1_step_y;\n') 
            f.write('                    pool1_kernel_layer(n)(pool_index) <= pool_in(n);\n')
            f.write('                end loop;\n\n')
        elif ct.feat_layers[l] == 'pool1':
            f.write("                p1_start <= '0';\n\n")
            
        f.write('                if {0}_offset_y < TO_UNSIGNED({1}_limit_y, {2}) then\n'.format(ct.feat_layers[l], ct.lays[l], ct.n_offset[l][1] + 1))    
        f.write('                    {0}_offset_y <= {0}_offset_y + TO_UNSIGNED({1}_step_y, {2});\n'.format(ct.feat_layers[l], ct.lays[l], ct.n_offset[l][1] + 1))
        for n in range(l, -1, -1):
            for d in range(len(dim)):
                if ct.feat_layers[n] != ct.feat_layers[l]:
                    f.write("                    {0}_offset_{1} <= (others => '0');\n".format(ct.feat_layers[n], dim[d]))    
        f.write('                    conv1_state <= OFF_RST;\n')
        
        f.write('                elsif {0}_offset_x < TO_UNSIGNED({1}_limit_x, {2})\n'.format(ct.feat_layers[l], ct.lays[l], ct.n_offset[l][0] + 1))
        f.write('                and {0}_offset_y = TO_UNSIGNED({1}_limit_y, {2}) then\n'.format(ct.feat_layers[l], ct.lays[l], ct.n_offset[l][1] + 1)) 
        f.write('                    {0}_offset_x <= {0}_offset_x + TO_UNSIGNED({1}_step_x, {2});\n'.format(ct.feat_layers[l], ct.lays[l], ct.n_offset[l][0] + 1)) 
        for n in range(l, -1, -1):
            for d in range(len(dim)):
                if ct.feat_layers[n] != ct.feat_layers[l] or (ct.feat_layers[n] == ct.feat_layers[l] and d != 0):
                    f.write("                    {0}_offset_{1} <= (others => '0');\n".format(ct.feat_layers[n], dim[d]))
        f.write('                    conv1_state <= OFF_RST;\n')
        
        f.write('                elsif {0}_offset_x = TO_UNSIGNED({1}_limit_x, {2})\n'.format(ct.feat_layers[l], ct.lays[l], ct.n_offset[l][0] + 1))
        f.write('                and {0}_offset_y = TO_UNSIGNED({1}_limit_y, {2}) then\n'.format(ct.feat_layers[l], ct.lays[l], ct.n_offset[l][1] + 1))
        if ct.feat_layers[l] == 'pool{0}'.format(ct.pool):
            f.write('                    conv1_state <= READ_FC{0}_DONE;\n'.format(ct.fc))
        else:
            f.write('                    conv1_state <= CHECK_{0};\n'.format(ct.feat_layers[l + 1].upper()))
        if ct.feat_layers[l] == 'conv1':    
            f.write("                    p1_start <= '1';\n")
        f.write('                end if;\n')
    f.write('            when READ_FC{0}_DONE =>\n'.format(ct.fc))
    f.write("                if fc{0}_done = '1' then\n".format(ct.fc))
    f.write('                    conv1_state <= WRITE_OUT;\n')
    f.write('                end if;\n')

    #State to write the output in the memmory
    f.write('            when WRITE_OUT =>\n')
    f.write('                o := TO_INTEGER(out_count);\n')
                
    f.write("                mem_we <= (others => '1');\n")
    f.write('                mem_addr <= STD_LOGIC_VECTOR(TO_UNSIGNED((start_bytes + width*width + 1)*nbytes+ o*nbytes, 32));\n')
    f.write('                mem_din <= STD_LOGIC_VECTOR(TO_SIGNED(TO_INTEGER(fc3_relu_layer(o)), 32));\n\n'.format())  
                    
    f.write('                if out_count < TO_UNSIGNED(fc{0}_neurons - 1,  {1}) then\n'.format(ct.fc, fn + 1))
    f.write('                    out_count <= out_count + TO_UNSIGNED(1, {0});\n'.format(fn + 1))    
    f.write('                elsif out_count = TO_UNSIGNED(fc{0}_neurons - 1, {1}) then\n'.format(ct.fc, fn + 1))
    f.write('                    conv1_state <= WRITE_DONE;\n')    
    f.write('                end if;\n')

    #State to write done signal in the memmory
    f.write('            when WRITE_DONE =>\n')
    f.write("                mem_addr <= (others => '0');\n")
    f.write('                mem_din <= x"DDDDDDDD";\n')
    f.write('                conv1_state <= DELAY;\n')

    #Write delay
    f.write('            when DELAY =>\n')
    f.write('                conv1_state <= BACK;\n')

    #State to back to another CNN computation
    f.write('            when BACK =>\n')
    f.write('                conv1_state <= INIT;\n\n')
                    
    f.write("                out_count <= (others => '0');\n")

    for l in range(len(ct.feat_layers)):
        for d in range(len(dim)):
            f.write("                {0}_offset_{1} <= (others => '0');\n".format(ct.feat_layers[l], dim[d]))


    f.write('            when OTHERS =>\n')
    f.write('                conv1_state <= RST_CONV;\n')
    f.write('        end case;\n\n')
            
            
    f.write('    end if;--End of reset\n')
    f.write('end process;\n')
    f.write('end Behavioral;\n')
    f.close()
    return
