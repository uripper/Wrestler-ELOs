from tqdm import trange
import random
import warnings
import numpy as np

# ignore runtime warnings
warnings.filterwarnings("ignore")

best_adj_rate_a = 152
best_const_match_a = 9
best_starting_elo_a = 665
best_adj_rate_b = 152
best_const_match_b = 9
best_starting_elo_b = 665
lowest_a=100
lowest_b=100
mse = 99999999999999999999999999999999999999999999999999999999999999999999999
# %%
import pandas as pd
df = pd.read_csv("Matches.csv")

df = df.dropna(subset=["Broadcast Date"])

for i in range(len(df)):
    if df["Match Winner(s)"][i] == "-":
        df.drop(i, inplace=True)

df.reset_index(drop=True, inplace=True)

roster_list = []

for i in df["Match Loser(s)"]:
    if i not in roster_list:
        roster_list.append(i)

for i in df["Match Winner(s)"]:
    if i not in roster_list:
        roster_list.append(i)

# %%
# Constants
for START_TEST in trange(1,2):
    for CONST_TEST in trange(1,1000000):
            mse_list = []

            ran_a = np.random.triangular(1, best_adj_rate_b, 3000)
            ran_a = round(ran_a)
            ran_b = np.random.triangular(1, best_const_match_b, 10)
            ran_b = round(ran_b)
            ran_c = np.random.triangular(1, best_starting_elo_b, 3000)
            ran_c = round(ran_c)
            adjustment_rate = ran_a
            const_match = ran_b
            starting_elo = ran_c

            # %%
            roster_dict = {}
            for i in roster_list:
                roster_dict[i] = starting_elo
                
            roster = pd.DataFrame.from_dict(roster_dict, orient="index")

            roster.columns = ["Elo"]

            roster.insert(1, "Number of Matches", 0)
            roster.insert(2, "Max Elo", starting_elo)
            roster.insert(3, "Min Elo", starting_elo)

            # %%
            adj_a_list = []
            adj_b_list = []
            outliers_dict = {"Match": []}

            # %%
            def find_adjustment(winner, loser, prob_winner, prob_loser, a_match, b_match, adjust=adjustment_rate, multicheck=False, i=0, const_match=const_match):
                l_length = 0
                adjust_a = 0
                adjust_b = 0
                if multicheck != False:
                    p = str(df["Competitors"][i])
                    q = winner.split(",")
                    z = loser.split(",")
                    if len(q) <= 1:
                        if len(z) > 1:
                            loser = z[0]


                        l = p.split(",")
                        for _ in range(100):
                            try:
                                l.remove(" ")
                            except ValueError:
                                break


                        l_length = len(l)
                        l_length = l_length

                if l_length > 2:
                    adjust_a = adjust * (1 - (prob_winner ** (1 / l_length)))
                    adjust_b = adjust / (1 - (prob_loser ** (1 / l_length)))

                if a_match < const_match:
                    adjust_a = (adjust)  * (const_match - a_match) * 2 + adjust_a

                else:
                    adjust_a = adjust + adjust_a


                if b_match < const_match:
                    adjust_b = adjust * (const_match - b_match) * 2 + adjust_b


                else:
                    adjust_b = adjust + adjust_b




                return adjust_a, adjust_b


            # %%
            def find_closest_roster(name):
                # Find lexical simularity of names
                if name in roster.index:
                    return roster.index[roster.index == name]
                elif roster.index.str.contains(name).any():
                    return roster.index[roster.index.str.contains(name)]
                else:
                    name = name.split(" ")
                    return [
                        roster.index[roster.index.str.contains(name[i])]
                        for i in range(len(name))
                    ]

            # %%
            def adjust_elo(winner, loser, elo_a=starting_elo, elo_b=starting_elo, a_match=0, b_match=0, i=0, multicheck=False, adjust=adjustment_rate, mse_list=mse_list, check_mse=False):

                prob_winner = 1 / (1 + 10 ** ((elo_b - elo_a) / adjust))
                prob_winner = .5
                prob_loser = 1 / (1 + 10 ** ((elo_a - elo_b) / adjust))
                prob_loser = .5


                adjust=adjustment_rate

                adjust_a, adjust_b = find_adjustment(winner, loser, prob_winner, prob_loser, a_match, b_match, i=i, multicheck=multicheck)
                adjust_a = (adjust_a) * (1 - prob_winner)
                adjust_b = (adjust_b) * (1 - prob_loser)
                adj_a_list.append(adjust_a)
                adj_b_list.append(adjust_b)
                elo_a = elo_a + (adjust_a) * (1 - prob_winner)
                elo_b = elo_b + (adjust_b) * (0 - prob_loser)
                elo_b = max(elo_b, 100)
                a_match += 1
                b_match += 1
                work_mse = prob_winner * (1 - prob_winner)

                if adjust_a > 300 or adjust_b > 300:
                    outliers_dict["Match"].append({"Winner": winner, "Loser": loser, "Adjustment": [adjust_a, adjust_b]})
                return elo_a, elo_b

            # %%
            def predict_winner(a, b):
                try:
                    elo_a = roster.loc[a, "Elo"]
                except:
                    new_a = find_closest_roster(a)
                    print(f"Did you mean the following/one of the following {new_a}?")
                    user_input = input("Enter the name of the roster you meant, or type 'new' for someone not on the roster: ")
                    if user_input == "new":
                        elo_a = starting_elo
                        a_match = 0
                    else:
                        elo_a = roster.loc[user_input, "Elo"]
                        a = user_input

                try:
                    elo_b = roster.loc[b, "Elo"]
                except:
                    new_b = find_closest_roster(b)
                    print(f"Did you mean the following/one of the following {new_b}?")
                    user_input = input("Enter the name of the roster you meant, or type 'new' for someone not on the roster: ")
                    if user_input != "new":
                        elo_b = roster.loc[user_input, "Elo"]
                        b = user_input
                    else:
                        elo_b = starting_elo
                        b_match = 0

                prob_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
                prob_b = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))
                try:
                    a_match = roster.loc[a, "Number of Matches"]
                except:
                    pass
                try:
                    b_match = roster.loc[b, "Number of Matches"]
                except:
                    pass

                adjust_a, adjust_x = find_adjustment(a, b, prob_a, prob_b, a_match, b_match)
                adjust_b, adjust_y = find_adjustment(b, a, prob_b, prob_a, b_match, a_match)

                adjust_a = adjust_a * (1 - prob_a)
                adjust_x = adjust_x * (0 - prob_b)

                adjust_b = adjust_b * (1 - prob_b)
                adjust_y = adjust_y * (0 - prob_a)

                print("-"*50)
                print(f"""
                    {a}'s Elo: {elo_a}
                    Probability of {a} winning: {round(prob_a*100, 2)}%. 
                    Rating change for win: {round(adjust_a)}.
                    New rating after win: {round(elo_a + adjust_a)}. 
                    Rating change for loss: {round(adjust_y)}.
                    New rating after loss: {round(elo_a + adjust_y)}.""")
                print(f"""
                    {b}'s Elo: {elo_b}
                    Probability of {b} winning: {round(prob_b*100, 2)}%.
                    Rating change for win: {round(adjust_b)}.
                    New rating after win: {round(elo_b + adjust_b)}.
                    Rating change for loss: {round(adjust_x)}.
                    New rating after loss: {round(elo_b + adjust_x)}.""")

            # %%
            x = 0
            biggest_upset = 0
            Winner = None
            Loser = None
            Initial_Winner_Elo = None
            Initial_Loser_Elo = None
            After_W_Elo = None
            After_L_Elo = None
            upsets_total = 0
            match_total = 0
            for i in range(len(df)):
                a = df["Match Winner(s)"][i]
                b = df["Match Loser(s)"][i]
                a_elo = roster.loc[a][0]
                b_elo = roster.loc[b][0]
                winner_elo, loser_elo = adjust_elo(a, b, a_elo, b_elo, roster.loc[a][1], roster.loc[b][1], i=i, multicheck=True, check_mse=True)
                roster.iloc[roster_list.index(a), 0] = winner_elo
                roster.iloc[roster_list.index(b), 0] = loser_elo
                roster.iloc[roster_list.index(a), 1] += 1
                roster.iloc[roster_list.index(b), 1] += 1
                
                # Change Winner's Max ELO and Loser's Min ELO if applicable
                
                if winner_elo > roster.loc[a][2]:
                    roster.iloc[roster_list.index(a), 2] = winner_elo
                
                if loser_elo < roster.loc[b][3]:
                    roster.iloc[roster_list.index(b), 3] = loser_elo
                
                if (loser_elo - winner_elo) > 50:
                    upsets_total += 1
                if (loser_elo - winner_elo) > biggest_upset:
                    biggest_upset = loser_elo - winner_elo
                    Winner = a
                    Loser = b
                    Initial_Winner_Elo = round(a_elo)
                    Initial_Loser_Elo = round(b_elo)
                    After_W_Elo = round(winner_elo)
                    After_L_Elo = round(loser_elo)
                match_total += 1

            correct = 0
            incorrect = 0
            for i in range(len(df)):
                a = df["Match Winner(s)"][i]
                b = df["Match Loser(s)"][i]
                if a.split(",")[0] == df["Match Winner(s)"][i]:
                    b = b.split(",")[0]
                a_elo = roster.loc[roster.index == a]["Elo"].values
                b_elo = roster.loc[roster.index == b]["Elo"].values
                prob_x = 1 / (1 + 10 ** ((b_elo - a_elo) / adjustment_rate))
                prob_y = 1 / (1 + 10 ** ((a_elo - b_elo) / adjustment_rate))
                new = 1- prob_x
                try:
                    mse_list.append(float(new))
                except:
                    new = .01
                    mse_list.append(new)

                if a_elo > b_elo:
                    correct += 1
                else:
                    incorrect += 1

            run_a = round(upsets_total/match_total * 100, 2)
            run_b = round(incorrect/(correct+incorrect)*100, 2)
            run_mse = np.mean(mse_list)**2*100
            sum_squared_error = sum(mse_list)
            if sum_squared_error < mse:
                print(f"\nNew best MSE: {run_mse}, Sum squared error: {np.sum(mse_list)**2}")
                print(f"Run A: {run_a}%, Run B: {run_b}%, adj_rate: {adjustment_rate}, starting_elo: {starting_elo}, const_match: {const_match}")
                mse = sum_squared_error
            if run_a < lowest_a and run_a > 2.5:
                lowest_a = run_a
                best_adj_rate_a = adjustment_rate
                best_const_match_a = const_match
                best_starting_elo_a = starting_elo
                print("\nNEW BEST")
                print(f"lowest_a: {lowest_a}, best_adj_rate_a: {best_adj_rate_a}, best_const_match_a: {best_const_match_a}, best_starting_elo_a: {best_starting_elo_a}")
                print(f"Run B: {run_b}%")
                print(f"Run MSE: {run_mse}")
                print(f"Sum squared error: {np.sum(mse_list)**2}")
            if run_b < lowest_b and run_b > 2.5:
                lowest_b = run_b
                best_adj_rate_b = adjustment_rate
                best_const_match_b = const_match
                best_starting_elo_b = starting_elo
                print("\nNEW BEST")
                print(f"lowest_b: {lowest_b}, best_adj_rate_b: {best_adj_rate_b}, best_const_match_b: {best_const_match_b}, best_starting_elo_b: {best_starting_elo_b}")
                print(f"Run A results: {run_a}%")
                print(f"Run MSE: {run_mse}")
                print(f"Sum squared error: {np.sum(mse_list)**2}")
print(f"""
      lowest_a: {lowest_a}
      best_adj_rate_a: {best_adj_rate_a}
      best_const_match_a: {best_const_match_a}
      best_starting_elo_a: {best_starting_elo_a}
      
      lowest_b: {lowest_b}
      best_adj_rate_b: {best_adj_rate_b}
      best_const_match_b: {best_const_match_b}
      best_starting_elo_b: {best_starting_elo_b}""")
                

