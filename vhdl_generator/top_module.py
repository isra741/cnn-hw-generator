import cnn_types as ct
import numpy as np

def gen_top_module():
    f = open(ct.vhdl_path + '\cnn.vhd', 'w')

    #Libraries includes
    f.write('library IEEE;\n')
    f.write('use IEEE.STD_LOGIC_1164.ALL;\n')
    f.write('use IEEE.NUMERIC_STD.ALL;\n')
    f.write('use ieee.std_logic_UNSIGNED.ALL;\n')
    f.write('use WORK.CNN_PACKAGE.ALL;\n\n')
    f.write('Library UNISIM;\n')
    f.write('use UNISIM.vcomponents.all;\n')
    f.write('Library UNIMACRO;\n')
    f.write('use UNIMACRO.vcomponents.all;\n\n')


    #Entity definition
    f.write('entity cnn is\n')
    f.write('    Port (clk : in STD_LOGIC;\n')
    f.write('          rst : in STD_LOGIC;\n')
    f.write('          mem_en : out STD_LOGIC;\n')
    f.write('          mem_we : out STD_LOGIC_VECTOR(3 downto 0);\n')
    f.write('          mem_dout : in STD_LOGIC_VECTOR(31 downto 0);\n')
    f.write('          mem_addr: out STD_LOGIC_VECTOR(31 downto 0);\n')
    f.write('          mem_din : out STD_LOGIC_VECTOR(31 downto 0));\n')
    f.write('end cnn;\n\n')

    f.write('architecture Behavioral of cnn is\n')

    #Conv control component declaration
    for l in range(ct.conv):
        la = l + 1
        f.write('\ncomponent conv{0}_control is\n'.format(la))
        f.write('Port (clk: in STD_LOGIC;\n')
        f.write('  rst_n: in STD_LOGIC;\n')
        f.write('  conv{0}_macc_out_layer: in CONV{1}_MACC_OUT_LAYER;\n\n'.format(la, la))
        if l== 0:
            f.write('  fc{0}_done : in STD_LOGIC;\n'.format(ct.fc))
            f.write('  fc{0}_relu_layer : in FC{1}_RELU_LAYER;\n'.format(ct.fc, ct.fc))
            f.write('  mem_dout : in STD_LOGIC_VECTOR(31 downto 0);\n')    
            f.write('  mem_addr : out STD_LOGIC_VECTOR(31 downto 0);\n')
            f.write('  mem_en : out STD_LOGIC;\n')
            f.write('  mem_din : out STD_LOGIC_VECTOR(31 downto 0);\n')
            f.write('  mem_we : out STD_LOGIC_VECTOR(3 downto 0);\n\n')
        else:
            f.write('  c{0}_start : in STD_LOGIC;\n'.format(la))
            f.write('  c{0}_in : in POOL{1}_KERNEL_CONV{2};\n'.format(la, la - 1, la))
            f.write('  c{0}_input : out CONV{1}_OUT;\n\n'.format(la, l))
               
        f.write('  c{0}_ce  : out STD_LOGIC;\n'.format(la))
        f.write('  c{0}_macc_rst : out STD_LOGIC;\n'.format(la))
        f.write('  c{0}_carryin : out STD_LOGIC;\n'.format(la))
        f.write('  c{0}_load  : out STD_LOGIC;\n'.format(la))
        f.write('  p{0}_start : out STD_LOGIC;\n'.format(la))
        f.write('  conv{0}_load_data : out CONV{1}_BIAS_LAYER;\n'.format(la, la))
        f.write('  conv{0}_weight_mux : out CONV{1}_WEIGHT_MUX;\n'.format(la, la))
        f.write('  pool{0}_kernel_layer : out POOL{1}_KERNEL_LAYER\n'.format(la, la))
        f.write('  );\n')
        f.write('end component conv{0}_control;\n'.format(la))

    #Pool control component declaration
    for l in range(ct.pool):
        la = l + 1
        f.write('\ncomponent pool{0}_control is\n'.format(la))
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
        f.write('end component pool{0}_control;\n'.format(la))
        
    #FC control component declaration    
    for l in range(ct.fc):
        la = l + 1
        f.write('\ncomponent fc{0}_control is\n'.format(la))
        f.write('    Port (clk : in STD_LOGIC;\n')
        f.write('          rst_n : in STD_LOGIC;\n')
        f.write('          fc{0}_start : in STD_LOGIC;\n'.format(la))       
        if l == 0:
            f.write('          fc{0}_in: in POOL{1}_IMG_FC{2};\n'.format(la, ct.pool, la))
        else:
            f.write('          fc{0}_in: in FC{1}_RELU_LAYER;\n'.format(la, la - 1))
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
            f.write('          fc{0}_input: out CONV{1}_OUT\n'.format(la, ct.pool))
        else:
            f.write('          fc{0}_input: out FC{1}_RELU\n'.format(la, la - 1))
        f.write('          );\n')
        f.write('end component fc{0}_control;\n'.format(la))

    #MAX Pool component declaration  
    for l in range(ct.pool):
        la = l + 1
        f.write('\ncomponent pool{0} is\n'.format(la))
        f.write('    Port ( clk : in STD_LOGIC;\n')
        f.write('           rst_n : in STD_LOGIC;\n')
        f.write('           enable : in STD_LOGIC;\n')
        f.write('           a : in POOL{0}_KERNEL;\n'.format(la))
        f.write('           y : out CONV{0}_OUT);\n'.format(la))
        f.write('end component pool{0};\n'.format(la))

    #Reset
    f.write('    --Reset conversion\n')
    f.write('    signal rst_n : STD_LOGIC;\n')
    f.write("    constant zeros : SIGNED({0} downto 0):= (others => '0');\n".format(ct.ncwf[0] - 1))

    #Conv MACC outputs signals declarations  
    for l in range(ct.conv):
        la = l + 1
        f.write('\n    --Convolution {0}  - MACC outputs\n'.format(la))
        for n in range(int(ct.conv_feat[l])):
            f.write('    signal conv{0}_macc_out{1}: CONV{2}_MACC_OUT;\n'.format(la, n, la))

    #Conv bias expansion 
    for l in range(ct.conv):
        la = l + 1
        f.write('\n    --Convolution {0}  - macc load\n'.format(la))
        for n in range(int(ct.conv_feat[l])):
            f.write('    signal conv{0}_macc_load{1} : std_logic_vector({2} downto 0);\n'.format(la, n, ct.ncoi[l] + ct.ncwf[l]*2))

    #Conv input expansion 
    for l in range(ct.conv):
        la = l + 1
        if l > 0:
            f.write('\n    --Convolution {0}  - input expansion\n'.format(la))
            f.write('    signal c{0}_input_ext : SIGNED({1} downto 0);\n'.format(la, ct.ncoi[l - 1] + ct.ncwf[l]))

    #Conv control signals
    for l in range(ct.conv):
        la = l + 1
        f.write('\n    --CONV {0} MACC configuration signals\n'.format(la))
        f.write('    signal c{0}_macc_rst: STD_LOGIC;\n'.format(la))
        f.write('    signal c{0}_carryin: STD_LOGIC;\n'.format(la))
        f.write('    signal c{0}_ce: STD_LOGIC;\n'.format(la))
        f.write('    signal c{0}_load: STD_LOGIC;\n'.format(la))
        f.write("    constant c{0}_addsub : STD_LOGIC := '1';\n".format(la))
        if l > 0:
            f.write('    signal c{0}_start : STD_LOGIC;\n'.format(la))
        
        f.write('    signal c{0}_done : STD_LOGIC;\n\n'.format(la))
        f.write('    --CONV {0} control output arrays\n'.format(la))
        f.write('    signal conv{0}_load_data : CONV{1}_BIAS_LAYER;\n'.format(la, la))
        f.write('    signal conv{0}_weight_mux : CONV{1}_WEIGHT_MUX;\n'.format(la, la))
        f.write('    signal pool{0}_kernel_layer: POOL{1}_KERNEL_LAYER;\n'.format(la, la))
        if l != 0:
            f.write('    signal c{0}_input: CONV{1}_OUT;\n\n'.format(la, l))
        else:
            f.write(' \n')
        f.write('    --CONV {0} control input arrays\n'.format(la))
        f.write('    signal conv{0}_macc_out_layer : CONV{1}_MACC_OUT_LAYER;\n'.format(la, la))

    #Pool out signals declarations  
    for l in range(ct.pool):
        la = l + 1
        f.write('\n    --Pool {0} outputs\n'.format(la))
        f.write('    signal p{0}_start: STD_LOGIC;\n'.format(la))
        for n in range(int(ct.conv_feat[l])):
            f.write('    signal pool{0}_out{1}: CONV{2}_OUT;\n'.format(la, n, la))
        f.write('\n    --Pool{0} control signals\n'.format(la))
        f.write('    signal pool{0}_pixel_layer: POOL{1}_PIXEL_LAYER;\n'.format(la, la))
        if l == (ct.pool - 1):
            f.write('    signal pool{0}_img_out: POOL{1}_IMG_FC{2};\n'.format(la, la, 1))
        else:
            f.write('    signal pool{0}_kernel_out: POOL{1}_KERNEL_CONV{2};\n'.format(la, la, ct.conv))
            
    #FC MACC output signals declarations  
    for l in range(ct.fc):
        la = l + 1
        f.write('\n    --Fully-connected {0} - MACC outputs\n'.format(la))
        for n in range(int(ct.fc_neurons[l])):
            f.write('    signal fc{0}_macc_out{1}: FC{2}_MACC_OUT;\n'.format(la, n, la))
     
    #FC MACC configuration signlas
    for l in range(ct.fc):
        la = l + 1
        f.write('\n    --FC {0} MACC configuration signals\n'.format(la))
        f.write('    signal fc{0}_macc_rst: STD_LOGIC;\n'.format(la))
        f.write('    signal fc{0}_carryin: STD_LOGIC;\n'.format(la))
        f.write('    signal fc{0}_ce: STD_LOGIC;\n'.format(la))
        f.write('    signal fc{0}_load: STD_LOGIC;\n'.format(la))
        f.write("    constant fc{0}_addsub : STD_LOGIC := '1';\n".format(la))
        f.write('    signal fc{0}_start : STD_LOGIC;\n\n'.format(la))
        
        f.write('    --FC {0} control output arrays\n'.format(la))
        f.write('    signal fc{0}_load_data : FC{1}_BIAS_LAYER;\n'.format(la, la))
        f.write('    signal fc{0}_weight_mux : FC{1}_WEIGHT_MUX;\n'.format(la, la))
        if l == 0:
            f.write('    signal fc{0}_input: CONV{1}_OUT;\n'.format(la, ct.pool))
        else:
            f.write('    signal fc{0}_input: FC{1}_RELU;\n'.format(la, la - 1))
        if l==(ct.fc - 1):
            f.write('    signal fc{0}_done : STD_LOGIC;\n\n'.format(la))
        f.write('    signal fc{0}_relu_layer : FC{1}_RELU_LAYER;\n'.format(la, la))
        f.write('    --FC {0} control input arrays\n'.format(la))
        f.write('    signal fc{0}_macc_out_layer : FC{1}_MACC_OUT_LAYER;\n'.format(la, la))

    #FC bias expansion  
    for l in range(ct.fc):
        la = l + 1
        f.write('\n    --Fully-connected {0} - macc load \n'.format(la))
        for n in range(int(ct.fc_neurons[l])):
            f.write('    signal fullyc{0}_macc_load{1} : std_logic_vector({2} downto 0);\n'.format(la, n, ct.nfoi[l] + ct.nfwf[l]*2))

    #FC input expansion  
    for l in range(ct.fc):
        la = l + 1
        f.write('\n    --Fully-connected {0} - input expansion \n'.format(la))
        if l == 0:
            f.write('    signal fc{0}_input_ext : SIGNED({1} downto 0);\n'.format(la, ct.ncoi[ct.conv - 1] + ct.nfwf[l]))
        else:
            f.write('    signal fc{0}_input_ext : SIGNED({1} downto 0);\n'.format(la, ct.nfoi[l - 1] + ct.nfwf[l]))

    f.write('begin\n')
    f.write('    rst_n <= not rst;\n')

    #Conv array signals instantiation 
    for l in range(ct.conv):
        la = l + 1
        f.write('\n    --Conv{0} out vector assignment\n'.format(la))
        for n in range(int(ct.conv_feat[l])):
            f.write('    conv{0}_macc_out_layer({1}) <= conv{2}_macc_out{3};\n'.format(la, n, la, n))
            
    #Pool array signals instantiation   
    for l in range(ct.pool):
        la = l + 1
        f.write('\n    --Pool{0} out vector assignment\n'.format(la))
        for n in range(int(ct.conv_feat[l])):
            f.write('    pool{0}_pixel_layer({1}) <= pool{2}_out{3};\n'.format(la, n, la, n))

    #FC array signals instantiation  
    for l in range(ct.fc):
        la = l + 1
        f.write('\n    --Fully-connected {0} out vector assignment\n'.format(la))
        for n in range(int(ct.fc_neurons[l])):
            f.write('    fc{0}_macc_out_layer({1}) <= fc{2}_macc_out{3};\n'.format(la, n, la, n))

    #Conv control component instantiation
    for l in range(ct.conv):
        la = l + 1
        f.write('\nCONV{0}_CTRL: conv{1}_control\n'.format(la, la))
        f.write('Port map (clk => clk,\n')
        f.write('  rst_n => rst_n,\n')
        f.write('  conv{0}_macc_out_layer => conv{1}_macc_out_layer,\n\n'.format(la, la))
        if l ==0:
            f.write('  fc{0}_done => fc{1}_done,\n'.format(ct.fc, ct.fc))
            f.write('  fc{0}_relu_layer => FC{1}_RELU_LAYER,\n'.format(ct.fc, ct.fc))
            f.write('  mem_dout => mem_dout,\n')    
            f.write('  mem_addr => mem_addr,\n')
            f.write('  mem_en => mem_en,\n')
            f.write('  mem_din => mem_din,\n')
            f.write('  mem_we => mem_we,\n\n')
        else:
            f.write('  c{0}_start => c{1}_start,\n'.format(la, la))
            f.write('  c{0}_in => pool{1}_kernel_out,\n'.format(la, l))
            f.write('  c{0}_input => c{1}_input,\n\n'.format(la, la))       
        f.write('  c{0}_ce  => c{1}_ce,\n'.format(la, la))
        f.write('  c{0}_macc_rst => c{1}_macc_rst,\n'.format(la, la))
        f.write('  c{0}_carryin =>  c{1}_carryin,\n'.format(la, la))
        f.write('  c{0}_load  =>  c{1}_load,\n'.format(la, la))
        f.write('  p{0}_start => p{1}_start,\n'.format(la, la))
        f.write('  conv{0}_load_data => conv{1}_load_data,\n'.format(la, la))
        f.write('  conv{0}_weight_mux => conv{1}_weight_mux,\n'.format(la, la))
        f.write('  pool{0}_kernel_layer => pool{1}_kernel_layer\n'.format(la, la))
        f.write('  );\n')

    #FC control component instantiation
    for l in range(ct.fc):
        la = l + 1
        f.write('\nFC{0}_CTRL: fc{1}_control\n'.format(la, la))
        f.write('Port map(clk => clk,\n')
        f.write('    rst_n =>  rst_n,\n')
        f.write('    fc{0}_start =>  fc{1}_start,\n'.format(la, la))   
        if l==0:
            f.write('    fc{0}_in=>  pool{1}_img_out,\n'.format(la, ct.pool))
        else:
            f.write('    fc{0}_in=>  fc{1}_relu_layer,\n'.format(la, l))
        f.write('    fc{0}_macc_out_layer => fc{1}_macc_out_layer,\n \n'.format(la, la))                     
        f.write('    fc{0}_ce=>  fc{1}_ce,\n'.format(la, la))
        f.write('    fc{0}_macc_rst=>  fc{1}_macc_rst,\n'.format(la, la))
        f.write('    fc{0}_carryin =>  fc{1}_carryin,\n'.format(la, la))
        f.write('    fc{0}_load =>  fc{1}_load,\n'.format(la, la))
        if l != (ct.fc - 1):
            f.write('    fc{0}_start =>  fc{1}_start,\n\n'.format(la + 1, la + 1))
        else:
            f.write('    fc{0}_done =>  fc{1}_done,\n\n'.format(la, la))
        f.write('    fc{0}_relu_layer => fc{1}_relu_layer,\n'.format(la, la))
        f.write('    fc{0}_load_data =>  fc{1}_load_data,\n'.format(la, la))
        f.write('    fc{0}_weight_mux =>  fc{1}_weight_mux,\n'.format(la, la))
        f.write('    fc{0}_input=>  fc{1}_input\n'.format(la, la))
        f.write('    );\n')

    #Pooling control component instantiation
    for l in range(ct.pool):
        la = l + 1
        f.write('\nPOOL{0}_CONTRL: pool{1}_control\n'.format(la, la))
        f.write('Port map (clk => clk,\n')
        f.write('    rst_n => rst_n,\n')
        f.write('    pool{0}_pixel_layer => pool{1}_pixel_layer,\n'.format(la, la))
        f.write('    p{0}_start  => p{1}_start,\n'.format(la, la))
        if l == 0:      
            f.write('    c{0}_start => c{1}_start,\n'.format(la + 1, la + 1))
        else:
            f.write('    fc{0}_start => fc{1}_start,\n'.format(1, 1))
        if l == (ct.pool - 1):
            f.write('    pool{0}_img_out => pool{1}_img_out\n'.format(la, la))
        else:
            f.write('    pool{0}_kernel_out => pool{1}_kernel_out\n'.format(la, la))
        f.write('    );\n')

    #Pool feat maps instantiation
    for l in range(ct.pool):
        la = l + 1
        for n in range(int(ct.conv_feat[l])):
            f.write('POOL{0}{1}: pool{2}\n'.format(la, n, la))
            f.write('Port map (clk => clk,\n')
            f.write('  rst_n => rst_n,\n')
            f.write('  enable => p{0}_start,\n'.format(la))
            f.write('  a => pool{0}_kernel_layer({1}),\n'.format(la, n))
            f.write('  y => pool{0}_out{1}\n'.format(la, n))
            f.write('  );\n')

    #CONV MACC instantiation
    for l in range(ct.conv):
        la = l + 1
        for n in range(int(ct.conv_feat[l])):
            f.write('C{0}{1}_DSP : MACC_MACRO\n'.format(la, n))
            f.write('      generic map (\n')
            f.write('         DEVICE => "7SERIES",  -- Target Device: "VIRTEX5", "7SERIES", "SPARTAN6"\n')
            f.write('         LATENCY => 1,         -- Desired clock cycle latency, 1-4\n')
            if l == 0:
                f.write('         WIDTH_A => {0},        -- Multiplier A-input bus width, 1-25\n'.format(1 + ct.pixel + ct.ncwf[l]))
            else:
                f.write('         WIDTH_A => {0},        -- Multiplier A-input bus width, 1-25\n'.format(1 + ct.ncoi[l - 1] + ct.ncwf[l]))
            f.write('         WIDTH_B => {0},        -- Multiplier B-input bus width, 1-18\n'.format(1 + ct.ncwf[l]))     
            f.write('         WIDTH_P => {0})        -- Accumulator output bus width, 1-48\n'.format(1 + ct.ncoi[l] + ct.ncwf[l]*2))
            f.write('      port map (\n')
            f.write('         SIGNED(P) => conv{0}_macc_out{1},     -- MACC ouput bus, width determined by WIDTH_P generic\n'.format(la, n))
            if l == 0:
                f.write('         A =>  mem_dout({0} downto 0),-- MACC input A bus, width determined by WIDTH_A generic\n'.format(ct.pixel + ct.ncwf[l]))
            else:
                f.write('         A =>  STD_LOGIC_VECTOR(c{0}_input_ext),-- MACC input A bus, width determined by WIDTH_A generic\n'.format(la))
            f.write('         ADDSUB => c{0}_ADDSUB, -- 1-bit add/sub input, high selects add, low selects subtract\n'.format(la))
            f.write('         B => std_logic_vector(conv{0}_weight_mux({1})),           -- MACC input B bus\n'.format(la, n)) 
            f.write('         CARRYIN => c{0}_carryin, -- 1-bit carry-in input to accumulator\n'.format(la))
            f.write('         CE => c{0}_CE,      -- 1-bit active high input clock enable\n'.format(la))
            f.write('         CLK => CLK,    -- 1-bit positive edge clock input\n')
            f.write('         LOAD => c{0}_LOAD, -- 1-bit active high input load accumulator enable\n'.format(la))
            f.write('         LOAD_DATA => conv{0}_macc_load{1}, -- Load accumulator input data,\n'.format(la, n)) 
            f.write('         RST => c{0}_macc_rst   -- 1-bit input active high reset\n'.format(la))
            f.write('      );\n')

    #Fully-connected MACC instantiation
    for l in range(ct.fc):
        la = l + 1
        for n in range(int(ct.fc_neurons[l])):
            f.write('FC{0}{1}_DSP : MACC_MACRO\n'.format(la, n))
            f.write('      generic map (\n')
            f.write('         DEVICE => "7SERIES",  -- Target Device: "VIRTEX5", "7SERIES", "SPARTAN6"\n')
            f.write('         LATENCY => 1,         -- Desired clock cycle latency, 1-4\n')
            if l == 0:
                f.write('         WIDTH_A => {0},        -- Multiplier A-input bus width, 1-25\n'.format(1 + ct.ncoi[ct.conv - 1] + ct.nfwf[l]))
            else:
                f.write('         WIDTH_A => {0},        -- Multiplier A-input bus width, 1-25\n'.format(1 + ct.nfoi[l - 1] + ct.nfwf[l]))
            f.write('         WIDTH_B => {0},        -- Multiplier B-input bus width, 1-18\n'.format(1 + ct.nfwf[l]))     
            f.write('         WIDTH_P => {0})        -- Accumulator output bus width, 1-48\n'.format(1 + ct.nfoi[l] + ct.nfwf[l]*2))
            f.write('      port map (\n')
            f.write('         SIGNED(P) => fc{0}_macc_out{1},     -- MACC ouput bus, width determined by WIDTH_P generic\n'.format(la, n)) 
            f.write('         A =>  std_logic_vector(fc{0}_input_ext),-- MACC input A bus, width determined by WIDTH_A generic\n'.format(la)) 
            f.write('         ADDSUB => fc{0}_ADDSUB, -- 1-bit add/sub input, high selects add, low selects subtract\n'.format(la))
            f.write('         B => std_logic_vector(fc{0}_weight_mux({1})),           -- MACC input B bus\n'.format(la, n)) 
            f.write('         CARRYIN => fc{0}_carryin, -- 1-bit carry-in input to accumulator\n'.format(la))
            f.write('         CE => fc{0}_CE,      -- 1-bit active high input clock enable\n'.format(la))
            f.write('         CLK => CLK,    -- 1-bit positive edge clock input\n')
            f.write('         LOAD => fc{0}_LOAD, -- 1-bit active high input load accumulator enable\n'.format(la))
            f.write('         LOAD_DATA => fullyc{0}_macc_load{1}, -- Load accumulator input data,\n'.format(la, n)) 
            f.write('         RST => fc{0}_macc_rst   -- 1-bit input active high reset\n'.format(la))
            f.write('      );\n')

    #CONV bias conversion
    for l in range(ct.conv):
        la = l + 1
        f.write(' \n')
        for n in range(int(ct.conv_feat[l])):
            f.write('    conv{0}_macc_load{1} <= STD_LOGIC_VECTOR(TO_SIGNED(TO_INTEGER(conv{2}_load_data({3})), \
    {4}));\n'.format(la, n, la, n, 1 + ct.ncoi[l] + ct.ncwf[l]*2))

    #CONV input shift
    f.write('\n')
    for l in range(ct.conv):
        la = l + 1
        if l > 0:
            
            f.write('    c{0}_input_ext <= c{0}_input & zeros;\n'.format(la))

    #FC bias conversion
    for l in range(ct.fc):
        la = l + 1
        f.write(' \n')
        for n in range(int(ct.fc_neurons[l])):
            f.write('    fullyc{0}_macc_load{1} <= STD_LOGIC_VECTOR(TO_SIGNED(TO_INTEGER(fc{2}_load_data({3})), \
    {4}));\n'.format(la, n, la, n, 1 + ct.nfoi[l] + ct.nfwf[l]*2))

    #FC input shift
    f.write('\n')
    for l in range(ct.fc):
        la = l + 1
        f.write('    fc{0}_input_ext <= fc{0}_input & zeros;\n'.format(la))

    f.write('end Behavioral;\n')
    f.close()
    return
