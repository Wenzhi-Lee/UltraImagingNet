from thop import profile, clever_format

def model_profile(model, input):
    flops, params = profile(model, inputs=input, verbose=True)
    macs, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {macs}, Params: {params}")