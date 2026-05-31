import os
from neuronguard import TabularClassifier

def main():
    print("====================================================================")
    print(" 🏉 NRL Team Composition: Raw Metrics Matchup Simulator 🏉")
    print("====================================================================\n")

    # Metrics: [Run Metres, Tackle Breaks, Try Assists, Errors, Tackles Made]
    # Classes: 0 (Disadvantage / Weak Link), 1 (Advantage / Star Player)
    classifier = TabularClassifier(
        num_classes=2, 
        num_features=5, 
        buckets_per_feature=10,
        amplify_delta=20,
        suppress_delta=5
    )

    # Dictionary mapping player names to their raw average metrics
    # [Metres, Breaks, Assists, Errors, Tackles]
    player_pool = {
        # Panthers
        "Nathan Cleary": [120.0, 3.0, 2.0, 0.0, 25.0],
        "Isaah Yeo": [150.0, 2.0, 1.0, 0.0, 45.0],
        "Cameron Munster": [140.0, 5.0, 1.0, 1.0, 20.0],
        "Inexperienced Rookie": [40.0, 0.0, 0.0, 3.0, 10.0],
        # Broncos
        "Reece Walsh": [200.0, 8.0, 2.0, 2.0, 5.0],
        "Payne Haas": [180.0, 5.0, 0.0, 0.0, 40.0],
        "Tired Forward": [50.0, 1.0, 0.0, 2.0, 15.0],
        "Fatigued Winger": [60.0, 1.0, 0.0, 3.0, 5.0]
    }

    # We train the classifier to recognize what "Good" and "Bad" metrics look like
    # Normally this would be thousands of rows of historical data
    training_data = [
        # Stars (1)
        player_pool["Nathan Cleary"] + [1],
        player_pool["Isaah Yeo"] + [1],
        player_pool["Cameron Munster"] + [1],
        player_pool["Reece Walsh"] + [1],
        player_pool["Payne Haas"] + [1],
        # Weak Links (0)
        player_pool["Inexperienced Rookie"] + [0],
        player_pool["Tired Forward"] + [0],
        player_pool["Fatigued Winger"] + [0],
    ]

    print("--- [Step 1] Training on Raw Player Metrics ---")
    classifier.fit(
        records=training_data,
        feature_indices=[0, 1, 2, 3, 4],
        label_index=5
    )
    print("Model Trained!\n")

    def calculate_team_advantage(team_name, players_list):
        total_advantage = 0
        print(f"{team_name} Roster:")
        for name in players_list:
            stats = player_pool[name]
            # predict_scores returns [Disadvantage_Score, Advantage_Score]
            scores = classifier.predict_scores(stats)
            advantage_contribution = scores[1]
            print(f"  - {name:<20} | Stats: {stats} | Power: {advantage_contribution}")
            total_advantage += advantage_contribution
            
        print(f" -> Total {team_name} Advantage Score: {total_advantage}\n")
        return total_advantage

    print("--- [Step 2] Initial Grand Final Matchup ---")
    team_panthers = ["Nathan Cleary", "Isaah Yeo", "Inexperienced Rookie"]
    team_broncos = ["Reece Walsh", "Payne Haas", "Tired Forward"]

    score_panthers = calculate_team_advantage("Penrith Panthers", team_panthers)
    score_broncos = calculate_team_advantage("Brisbane Broncos", team_broncos)

    print(f"🏆 INITIAL PREDICTION: {'Panthers' if score_panthers > score_broncos else ('Broncos' if score_broncos > score_panthers else 'Tie')} have the advantage!\n")

    print("--- [Step 3] Mid-Game Injury / Interchange! ---")
    print("   Panthers take off 'Inexperienced Rookie' and bring on 'Cameron Munster'!")
    print("   Broncos replace 'Tired Forward' with 'Fatigued Winger'!\n")

    team_panthers = ["Nathan Cleary", "Isaah Yeo", "Cameron Munster"]
    team_broncos = ["Reece Walsh", "Payne Haas", "Fatigued Winger"]

    score_panthers = calculate_team_advantage("Penrith Panthers", team_panthers)
    score_broncos = calculate_team_advantage("Brisbane Broncos", team_broncos)

    print(f"🏆 NEW PREDICTION: {'Panthers' if score_panthers > score_broncos else ('Broncos' if score_broncos > score_panthers else 'Tie')} have the massive advantage!\n")

    print("--- [Step 4] The Magic of Hebbian Accumulation ---")
    print("   Because NeuronGuard doesn't use rigid fixed-length dense arrays, ")
    print("   we can linearly aggregate (sum) the associative potentials of ")
    print("   an arbitrary number of independent players to get a total team score! ")
    print("   This makes dynamically calculating team trades instantaneous.")
    print("====================================================================")

if __name__ == "__main__":
    main()
