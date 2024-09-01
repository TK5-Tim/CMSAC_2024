library(arrow)
library(ggplot2)
library(dplyr)
library(tidyr)
library(devtools)
install_local('/Users/cincy/OneDrive/Documents/Sam R Work/BDBCleanR')
library(BDBCleanR)

players<-read.csv("players.csv")
#players<-add_player_identifiers(players,position_name="officialPosition")
plays<-read.csv('plays.csv')
games<-read.csv("games.csv")
pffData<-read.csv("pffScoutingData.csv")

pres_df<-read_parquet('full_pressure_preds.parquet')
tar_df<-read_parquet('full_target_preds.parquet')
comp_df<-read_parquet('full_completion_preds.parquet')
comp_input_df<-read_parquet('target_comp_model_df.parquet')


weekly_data<-data.frame()

for (i in 1:8){
  wk<-read.csv(paste0("week",i,".csv"))
  wk$week<-i
  weekly_data<-rbind(weekly_data,wk)
}

wk<-""


str(weekly_data)


wk_std <- standardize_data(weekly_data)

wk_std <- wk_std %>%
  group_by(gameId, playId) %>%
  mutate(snap_time = second[event == "ball_snap"][1],
         second_since_snap = second - snap_time) %>%
  ungroup()%>%select(-snap_time)


### We need a row of pressure probabilities, plus receiver and coverage data

pressure_df <- pres_df %>%
  select(gameId, playId, frameId, nflId, pressure_prob) %>%
  group_by(gameId, playId, frameId) %>%                       # Group by gameId, playId, and frameId
  arrange(desc(pressure_prob), .by_group = TRUE) %>%          # Arrange by pressure_prob in descending order
  mutate(player_dist_rank = row_number()) %>%                 # Rank based on pressure_prob within the group
  ungroup()                                                   # Ungroup the data



# Step 1: Group by gameId, playId, frameId and rank players by pressure_prob
pressure_df_ranked <- pressure_df %>%
  group_by(gameId, playId, frameId) %>%
  arrange(desc(pressure_prob), .by_group = TRUE) %>%
  mutate(rank = row_number()) %>%
  ungroup()

# Step 2: Pivot the data to wide format
wide_df <- pressure_df_ranked %>%
  pivot_wider(
    id_cols = c(gameId, playId, frameId),  # Specify the columns to use as identifiers
    names_from = rank,
    values_from = c(nflId, pressure_prob),
    names_glue = "{.value}_{rank}"
  ) %>%
  mutate(across(starts_with("nflId_"), ~replace_na(., 0)),
         across(starts_with("pressure_prob_"), ~replace_na(., 0)))

# View the resulting wide format data frame
head(wide_df)%>%View()

full_comp_model_df<-tar_df%>%left_join(wide_df,by=c('gameId','playId','frameId'))%>%filter(frameId>5)


tracking_data_df<-wk_std%>%select(gameId,playId,frameId,event,second_since_snap)%>%distinct()


full_comp_model_df<-full_comp_model_df%>%
left_join(tracking_data_df,by=c('gameId','playId','frameId'))%>%
left_join(comp_input_df%>%select(gameId,playId,frameId,nflId,incomplete_pass),by=c('gameId','playId','frameId','nflId'))



write_parquet(full_comp_model_df, paste0('full_comp_model_df','.parquet'))


