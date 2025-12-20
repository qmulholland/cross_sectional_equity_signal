"""
Currently not being used. It is being kept in as a possible 'next step'
changing from decile logic towards proportional trades by signal strength.

Portfolio.py calculates a signal-proportional weighting for each assset in the
dataset by dividing each individual signal by the sum of all absolute signals
for that date. It then appends these weights to a new column, ensuring that the
total portfolio exposure (long and short combined) is equal to 1.0 (100%).
"""

import pandas as pd     #imports pandas


def equal_weight_portfolio(df: pd.DataFrame, signal_col: str):      #Groups the data by date, weighting is  
                                                                    #done independently for each day

    weights = (         
        df.groupby("date")[signal_col]                  #Groups by date
        .transform(lambda x: x / x.abs().sum())         # - Uses transform to calculate the weight of each
                                                        #   asset relative to the total day's signals

                                                        # - lambda x: x / x.abs().sum() ensures sum of absolute
                                                        #   weights equals 1.0 (100% Exposure)
    )
    df = df.copy()             #Creates a copy of the DataFrame to avoid modifying original input data

    df["weight"] = weights     #Assigns the calculated weights to a new column named "weight"

    return df          #Returns the updated DataFrame containing new weight column
