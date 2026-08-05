from assets.registry import load_assets, load_categories, grouped_assets

def test_registry_is_grouped_and_complete():
    assets = load_assets()
    categories = load_categories()
    groups = grouped_assets()
    assert len(assets) >= 40
    assert set(groups) == set(categories)
    assert sum(len(items) for items in groups.values()) == len(assets)
    assert all(asset.cftc_code and asset.price_symbol for asset in assets.values())
