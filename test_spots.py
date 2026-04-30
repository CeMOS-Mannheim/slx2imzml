import scilslab as sl
path = "D:/mmi/PPS_S10_run.slx"
with sl.LocalSession(path) as slx_file:
    dataset = slx_file.dataset_proxy
    region_tree = dataset.get_region_tree()
    all_regions = region_tree.get_all_regions()
    for r in all_regions:
        if len(r.subregions) == 0:
            spots = dataset.get_region_spots(r.id)
            print(f"Region: {r.name}, Type: {type(spots)}")
            if isinstance(spots, dict):
                print(f"Keys: {list(spots.keys())}")
                for k, v in spots.items():
                    print(f"Key {k}: Type {type(v)}")
                    try:
                        print(f"  Length: {len(v)}")
                    except:
                        pass
            break
