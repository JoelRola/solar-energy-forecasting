def test_public_package_and_model_imports():
    import solar_forecasting
    from solar_forecasting.models import blstm, cnn, lstm, lstm_cnn
    from solar_forecasting.models import random_forest, xgboost_model

    assert solar_forecasting.__version__
    assert all(
        callable(module.build_model)
        for module in (blstm, cnn, lstm, lstm_cnn, random_forest, xgboost_model)
    )


def test_synthetic_sample_loads():
    from solar_forecasting.config import SAMPLE_DATA_PATH
    from solar_forecasting.data import load_processed_data
    from solar_forecasting.sequences import make_sequence_split

    frame = load_processed_data(SAMPLE_DATA_PATH)
    split = make_sequence_split(frame, lookback=24)
    assert len(split.X_train) > 0
    assert len(split.X_test) > 0
