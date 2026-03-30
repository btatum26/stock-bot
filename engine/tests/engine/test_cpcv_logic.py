import numpy as np
from engine.core.optimization.cpcv_splitter import CPCVSplitter

def test_purge_logic():
    splitter = CPCVSplitter()
    
    # Simple setup: train is 0-79, val is 80-99
    train_idx = np.arange(80)
    val_idx = np.arange(80, 100)
    l_max = 5
    
    purged_train = splitter._apply_purge_protocol(train_idx, val_idx, l_max)
    
    # Expected: 75, 76, 77, 78, 79 are purged (5 rows before 80)
    assert len(purged_train) == 75
    assert 74 in purged_train
    assert 75 not in purged_train
    assert purged_train[-1] == 74

def test_embargo_logic():
    splitter = CPCVSplitter()
    
    # Setup: val is 0-19, train is 20-99 (training after validation)
    val_idx = np.arange(20)
    train_idx = np.arange(20, 100)
    pct = 0.1 # Using 10% for easier testing, default is 0.01
    
    # Dataset size is 100. 10% embargo = 10 rows.
    # Validation ends at 19. Embargo should remove indices 20-29.
    embargoed_train = splitter._apply_embargo_protocol(train_idx, val_idx, pct=pct)
    
    # Expected: 20-29 are removed. Train starts at 30.
    assert len(embargoed_train) == 70 # 80 total - 10 removed
    assert 29 not in embargoed_train
    assert 30 in embargoed_train
    assert embargoed_train[0] == 30
