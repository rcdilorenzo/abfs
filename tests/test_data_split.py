from src.group_data_split import GroupDataSplit, DataSplitConfig

def test_total(sample_df):
    split = GroupDataSplit(sample_df, key='group_id')
    assert split.total == 10

def test_train_size(sample_df):
    # 30% training data, 3 of 10 items
    config = DataSplitConfig(0.5, 0.2, 0)

    split = GroupDataSplit(sample_df, key='group_id', config=config)
    print(split.train_df())
    assert len(split.train_df()) == 3

def test_validation_size(sample_df):
    # 50% validation data, 5 of 10 items
    config = DataSplitConfig(0.5, 0.2, 0)

    split = GroupDataSplit(sample_df, key='group_id', config=config)
    assert len(split.val_df) == 5

def test_test_size(sample_df):
    # 20% test data, 2 of 10 items
    config = DataSplitConfig(0.5, 0.2, 0)

    split = GroupDataSplit(sample_df, key='group_id', config=config)
    assert len(split.test_df) == 2

def test_unique_groups(sample_df):
    split = GroupDataSplit(sample_df, key='group_id')

    train_df = split.train_df()
    val_df = split.val_df
    test_df = split.test_df

    assert train_df.group_id.isin(test_df.group_id).any() == False
    assert train_df.group_id.isin(val_df.group_id).any() == False

    assert test_df.group_id.isin(train_df.group_id).any() == False
    assert test_df.group_id.isin(val_df.group_id).any() == False

    assert val_df.group_id.isin(test_df.group_id).any() == False
    assert val_df.group_id.isin(train_df.group_id).any() == False

def test_not_shuffling_test_set_after_initial_load(sample_df):
    split = GroupDataSplit(sample_df, key='group_id')

    assert (split.test_df.value == split.test_df.value).all()

def test_not_shuffling_validation_set_after_initial_load(sample_df):
    split = GroupDataSplit(sample_df, key='group_id')

    assert (split.val_df.value == split.val_df.value).all()

def test_shuffling_training_set(sample_df):
    split = GroupDataSplit(sample_df, key='group_id')

    assert (split.train_df().value != split.train_df().value).any()
