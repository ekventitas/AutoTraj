import pyarrow.parquet as pq

table = pq.read_table("/hy-tmp/Tool-Star/Tool_Star_RL/mix_grpo/grpo_mix_train_shuffle.parquet")
#table = pq.read_table("/hy-tmp/Tool-Star/Tool_Star_RL/mix_grpo/grpo_mix_test.parquet")
pq.write_table(table.slice(5760,10000), "/hy-tmp/Tool-Star/Tool_Star_RL/mix_grpo/grpo_train_4_remain.parquet")
#pq.write_table(table.slice(0,150), "/hy-tmp/Tool-Star/Tool_Star_RL/mix_grpo/grpo_test_150.parquet")
