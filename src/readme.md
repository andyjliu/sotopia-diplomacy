### Examples

<!-- Generate Episode from Environment Profile tag -->
python generate_episode.py --env_tag random_sample_100_games --epi_tag random_sample_100_games --sub_sample_begin 0 --sub_sample_end None

<!-- Get the value after countries movement with dialogue -->
python intent_value_evaluate.py \
--res_path data/intent_response/intent_responses_whole.json \
--tgt_path data/results/countries_value_move_with_dialogue.json \
--move 


<!-- Get the value after countries movement without dialogue -->
python intent_value_evaluate.py \
--res_path data/intent_response/intent_responses_whole_without_dialogue.json \
--tgt_path data/results/countries_value_move_without_dialogue.json \
--move 

<!-- Get the value at the end of current phase, without movement from countries -->
python intent_value_evaluate.py \
--res_path data/intent_response/intent_responses_whole_without_dialogue.json \
--tgt_path data/results/countries_value_move_without_movement.json \

<!-- Get the value at the end of previous phase, without movement -->
python intent_value_evaluate.py \
--res_path data/intent_response/intent_responses_whole_without_dialogue.json \
--tgt_path data/results/countries_value_previous_phase.json \
--prev

<!-- Get the value at the end of current phase, with real moves from countries -->
python intent_value_evaluate.py \
--res_path data/intent_response/intent_responses_whole_without_dialogue.json \
--tgt_path data/results/countries_value_current_phase.json \
--real

<!-- Store EnvironmentProfile Tags -->
1. random_sample_100_games: This is the stored environment profile for 100 randomly picked samples, with cleaning the whole global message and only contains 1903-1906 phases without non Movement phases.
2. ntaske
3. taskeval_n

<!-- Store EpisodeLog Tags -->
1. random_sample_100_games: Generate 775 episodes from the 'random_sample_100_games' environment profiles.
(On going: Finished 0-400 (Llama-3-70B), rest of them are 400 - 775)
2. ntaske
3. taskeval_n




Message num correlation with phases
{2: 3350,
 3: 2506,
 4: 1826,
 5: 1361,
 6: 1011,
 7: 749,
 8: 551,
 9: 425,
 10: 334,
 11: 260}