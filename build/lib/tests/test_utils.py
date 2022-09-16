from art_api.utils import init, load_data
import pandas as pd

def test_init():
    assert len(init()) == 2
    imgs, df = init()
    assert type(imgs) == list
    assert type(df) == pd.DataFrame
    return imgs, df
    
def test_load_data():
    X, y = load_data(df)
    assert X.shape[0] == y.shape[0]