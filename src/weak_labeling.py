from snorkel.labeling import LabelingFunction, labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel import labeling as lbl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('features/features_january.csv')
df["ts_event"] = pd.to_datetime(df["ts_event"])
df = df.sort_values("ts_event").reset_index(drop=True)

ABSTAIN = -1
LEGIT = 0
SPOOF = 1

# Old Labeling-Functions
'''
# Labeling functions for weak labels
@labeling_function()
def large_and_distance_and_cancel(df):

    LARGE_ORDER = 3

    if (
        df["distance_ticks"] >= 10 and df["distance_ticks"] < 40 and
        df["ended_with_cancel"] == 1 and
        df["relative_size"] >= LARGE_ORDER
    ):
        return SPOOF
    elif (
        df["ended_with_cancel"] == 0 or
        df["distance_ticks"] < 3
    ):
        return LEGIT
    return ABSTAIN

@labeling_function()
def large_shortlifetime(df):

    LARGE_ORDER = 3
    SHORT_LIFETIME = 600

    if (
        df["relative_size"] >= LARGE_ORDER and
        np.expm1(df["log_lifetime"]) < SHORT_LIFETIME and
        df["ended_with_cancel"] == 1
    ):
        return SPOOF
    elif (
        df["ended_with_cancel"] == 0 and
        df["log_lifetime"] > 10
    ):
        return LEGIT
    return ABSTAIN

@labeling_function()
def price_attraction_and_reversion(df):

    PRICE_REVERSION = 0
    ATTRACTION_RATIO = 0.05

    if(
        abs(df["price_reversion"]) > PRICE_REVERSION and
        abs(df["attraction_ratio"]) > ATTRACTION_RATIO
    ):
        return SPOOF
    return ABSTAIN

@labeling_function()
def depth_wall(df):

    LARGE_ORDER = 3
    DEPTH_RATIO = 3

    if ( 
        df["relative_size"] >= LARGE_ORDER and
        df["depth_ratio"] > DEPTH_RATIO
    ): 
        return SPOOF
    return ABSTAIN

@labeling_function()
def aggressive_liquidity(df):

    LARGE_ORDER = 3
    DISTANCE = 3
    POST_TRADE_COUNT = 0

    if (
        df["relative_size"] >= LARGE_ORDER and
        df["distance_ticks"] <= DISTANCE and
        df["post_trade_count"] > POST_TRADE_COUNT
    ):
        return LEGIT
    return ABSTAIN

@labeling_function()
def large_longlifetime(df):
    
    LARGE_ORDER = 3
    LONG_LIFETIME = 4000

    if (
        df["relative_size"] >= LARGE_ORDER and
        np.expm1(df["log_lifetime"]) >= LONG_LIFETIME
    ):
        return LEGIT
    return ABSTAIN

@labeling_function()
def imbalance_shock_after_cancel(df):
    
    OBI_SHIFT = 0.35
    LARGE_ORDER = 3

    if (
        abs(df["order_book_imbalance_shift"]) >= OBI_SHIFT and
        df["relative_size"] >= LARGE_ORDER and
        df["ended_with_cancel"] == 1
    ):
        return SPOOF
    return ABSTAIN

@labeling_function()
def no_depth_impact(df):

    LARGE_ORDER = 2
    DEPTH_VOLUME = 2000

    if (
        df["relative_size"] >= LARGE_ORDER and
        df["depth_volume"] >= DEPTH_VOLUME
    ):
        return LEGIT
    return ABSTAIN

@labeling_function()
def post_trade_price_reversion(df):

    POST_TRADE_COUNT = 7
    PRICE_REVERSION = 0

    if (
        df["post_trade_count"] >= POST_TRADE_COUNT and
        abs(df["price_reversion"]) > PRICE_REVERSION
    ): 
        return SPOOF
    return ABSTAIN

@labeling_function()
def cancel_burst_and_midprice_change(df):

    CANCEL_BURST = 4
    MIDPRICE_CHANGE = 0.025

    if (
        df["post_cancel_count"] >= CANCEL_BURST and
        abs(df["midprice_change_1000ms"]) >= MIDPRICE_CHANGE
    ):
        return SPOOF
    return ABSTAIN


labeling_list = [
    large_and_distance_and_cancel, large_shortlifetime, price_attraction_and_reversion, 
    depth_wall, aggressive_liquidity, large_longlifetime,
    imbalance_shock_after_cancel, no_depth_impact, post_trade_price_reversion, cancel_burst_and_midprice_change
    ]
'''

# Labeling functions
@labeling_function()
def quick_cancel(df):
    # Any order that is cancelled quickly --> non-genuine order
    if (
        df["ended_with_cancel"] == 1 and
        np.expm1(df["log_lifetime"]) < 300
    ):
        return SPOOF
    elif np.expm1(df["log_lifetime"]) > 4000:
        return LEGIT
    return ABSTAIN

@labeling_function()
def extremely_quick_cancel(df):
    # Voting amplification for even quicker cancel
    if (
        df["ended_with_cancel"] == 1 and
        np.expm1(df["log_lifetime"]) < 100
    ):
        return SPOOF
    return ABSTAIN

@labeling_function()
def price_reverted_after_end(df):
    # Strong price reversion indicates spoofing
    if df["price_reversion"] < -0.02:
        return SPOOF
    elif df["price_reversion"] > 0.02:
        return LEGIT
    return ABSTAIN

@labeling_function()
def signed_impact_reversal(df):
    # Looks at the price impact without reversion, but instead from view of the spoofing order after cancel
    if df["signed_impact_after"] < -2.0:
        return SPOOF
    elif df["signed_impact_after"] > 2.0:
        return LEGIT
    return ABSTAIN

@labeling_function()
def imbalance_shock_on_cancel(df):
    # Book imbalance shifted significantly after cancellation
    if (
        abs(df["order_book_imbalance_shift"]) >= 0.4 and
        df["ended_with_cancel"] == 1
    ):
        return SPOOF
    return ABSTAIN

@labeling_function()
def traded_against(df):
    # Order resulted in trades and was not cancelled is most likely genuine market making
    if (
        df["post_trade_count"] > 0 and
        df["ended_with_cancel"] == 0
    ):
        return LEGIT
    return ABSTAIN

@labeling_function()
def cancel_burst_with_price_move(df):
    # Cancel burst is a sign that traders have been manipulated
    if (
        df["post_cancel_count"] >= 4 and
        df["ended_with_cancel"] == 1 and
        abs(df["midprice_change_1000ms"]) >= 0.03
    ):
        return SPOOF
    return ABSTAIN

@labeling_function()
def attraction_then_reversal(df):
    # Significant attraction AND reversion is a clear spoofing sign
    if (
        df["attraction_ratio"] > 0.08 and
        df["price_reversion"] < 0
    ):
        return SPOOF
    return ABSTAIN

@labeling_function()
def spread_widened_on_cancel(df):
    # Spread widening can be a spoofing sign
    if (
        df["spread_change_ticks"] >= 3 and
        df["ended_with_cancel"] == 1
    ):
        return SPOOF
    return ABSTAIN

@labeling_function()
def long_lived_passive(df):
    # Long-lived orders with no impact are legitimate
    if (
        np.expm1(df["log_lifetime"]) >= 4000 and
        abs(df["midprice_change_during"]) < 0.01 and
        df["ended_with_cancel"] == 0
    ):
        return LEGIT
    return ABSTAIN

@labeling_function()
def no_post_impact(df):
    # No price movement and no cancel are genuine orders part of the trading flow
    if (
        abs(df["midprice_change_1000ms"]) < 0.005 and
        df["ended_with_cancel"] == 0
    ):
        return LEGIT
    return ABSTAIN

@labeling_function()
def strong_signed_impact(df):
    # Very strong directional price impact after order ended
    if df["signed_impact_after"] < -3.0:
        return SPOOF
    elif df["signed_impact_after"] > 3.0:
        return LEGIT
    return ABSTAIN

@labeling_function()
def cancel_with_imbalance_and_price_move(df):
    # Combined signal of imbalance shift and midprice shift
    if (
        df["ended_with_cancel"] == 1 and
        abs(df["order_book_imbalance_shift"]) >= 0.3 and
        abs(df["midprice_change_200ms"]) >= 0.01
    ):
        return SPOOF
    return ABSTAIN

@labeling_function()
def passive_with_trade_and_no_reversion(df):
    # Legitimate signal combination
    if (
        df["post_trade_count"] > 0 and
        df["ended_with_cancel"] == 0 and
        df["price_reversion"] > 0
    ):
        return LEGIT
    return ABSTAIN

labeling_list = [
    quick_cancel,
    extremely_quick_cancel,
    price_reverted_after_end,
    signed_impact_reversal,
    imbalance_shock_on_cancel,
    traded_against,
    cancel_burst_with_price_move,
    attraction_then_reversal,
    spread_widened_on_cancel,
    long_lived_passive,
    no_post_impact,
    strong_signed_impact,
    cancel_with_imbalance_and_price_move,
    passive_with_trade_and_no_reversion,
]

applier = lbl.PandasLFApplier(labeling_list)
label_matrix = applier.apply(df)

# Analysing feature coverage
print(lbl.LFAnalysis(label_matrix).label_coverage())
print(lbl.LFAnalysis(label_matrix).label_conflict())
print(lbl.LFAnalysis(label_matrix).label_overlap())
print(lbl.LFAnalysis(label_matrix).lf_summary())

# Label Model and applying probabilites to features_df
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=label_matrix, n_epochs=5000, log_freq=100)

y_prob = label_model.predict_proba(label_matrix).astype(float)
df["spoof_prob"] = y_prob[:,1]
HIGH = 0.85
LOW = 0.35
df["weak_label"] = -1
df.loc[df["spoof_prob"] >= HIGH, "weak_label"] = 1
df.loc[df["spoof_prob"] <= LOW, "weak_label"] = 0

print(df['weak_label'].value_counts())
train_df = df[df["weak_label"] != -1].copy()

print(train_df.describe())
print(train_df['weak_label'].value_counts())

ax = df["spoof_prob"].hist(bins=50)
fig = ax.get_figure()
fig.savefig('features/spoofprob_distri.pdf')

train_df.to_csv('data/Training and Testing/test_january.csv')