library(arrow)
library(ggplot2)
library(dplyr)

players<-read.csv("players.csv")
#players<-add_player_identifiers(players,position_name="officialPosition")
plays<-read.csv('plays.csv')
games<-read.csv("games.csv")
pffData<-read.csv("pffScoutingData.csv")

pres_df<-read_parquet('full_pressure_preds.parquet')
tar_df<-read_parquet('full_target_preds.parquet')
comp_df<-read_parquet('full_completion_preds.parquet')

str(tar_df)
str(comp_df)



# Assuming your data frame is named `df`
tar_df <- tar_df %>%
  group_by(gameId, playId, frameId) %>%            # Step 1: Group by gameId, playId, and frameId
  arrange(desc(target_prob), .by_group = TRUE) %>% # Step 2: Arrange by target_prob in descending order within each group
  mutate(target_rank = row_number()) %>%           # Step 3: Create target_rank column by ranking within the group
  ungroup()                                        # Ungroup the data


tar_df%>%select(target_prob,target_rank,is_target,frameId)%>%View()