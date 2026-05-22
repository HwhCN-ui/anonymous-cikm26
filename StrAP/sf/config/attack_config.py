attack = {
    "poison_intensity": 2.5,
}

backdoor = {
    "enabled": False,
    
    "trigger_type": "badnets",

    "target_label": 1,
    "poison_ratio": 0.5,

    "trigger_size": 4,          
    "trigger_value": 1.0,      
    "trigger_position": "bottom_right", 


    "tact_seed": 0,
    "tact_enable_cover": True,

    "blended_alpha": 0.2,
    "blended_seed": 0,

    "pfedba_alpha": 0.2,
}

rank_aware = {
    "proj_strength": 2.0,
}

alie = {
    "z": 2.5,
    "sign": -1,
    "eps": 1e-9,
}

minmax = {
    "direction": "neg_mean",     
    "gamma_cap": 1e9,
    "gamma_scale": 1.5,
    "nonneg": True,
    "eps": 1e-12,

    "align_mix": 0.0,
    "cos_min": 0.0,
    "tau_norm": None,
}

minsum = {
    
}

poisonedfl = {
    "c_init": 8.0,
    "c_min": 0.5,
    "decay": 0.7,
    "eps": 1e-12,
    "p_thresh": 0.01,
    "cos_min": 0.0,
    "warmup_alpha": 1e-3,
}

signflip = {
    "gamma": 2.5,
}

scaling = {
    "lambda": 2.5,
}

gaussian = {
    "std_scale": 2.0,
    "eps": 1e-12,
}

neurotoxin = {
    "keep_ratio": 0.2,        
    "score_source": "benign", 
    "rescale": True,          
    "eps": 1e-12,
}

cerp = {
    "radius_scale": 1.0,      
    "mix_with_benign": 0.0,   
    "clip_norm": None,        
    "eps": 1e-12,
}

pfedba = {
    "trigger_opt": True,
    "trigger_steps": 1,
    "trigger_lr": 0.1,
    "tv_weight": 1e-4,
    "l2_weight": 1e-4,
    "random_position": False,  

    "update_scale": 1.0,        
    "stealth_projection": True,
    "radius_scale": 1.0,        
    "clip_norm": None,
    "eps": 1e-12,
}