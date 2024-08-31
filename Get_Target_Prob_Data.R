library(devtools)
library(nflfastR)
library(nflreadr)
library(dplyr)
install_local('/Users/cincy/OneDrive/Documents/Sam R Work/BDBCleanR')
library(BDBCleanR)
library(ggplot2)
library(gganimate)
library(tidyr)
library(arrow)


### Distance Function Calculation
euclidean_distance <- function(x1, y1, x2, y2) {
  sqrt((x1 - x2)^2 + (y1 - y2)^2)
}
  
  
is_oriented_towards <- function(qb_x, qb_y, qb_orientation, receiver_x, receiver_y, threshold = 30) {
    # Calculate the vector from QB to receiver
    vector_x <- receiver_x - qb_x
    vector_y <- receiver_y - qb_y
    
    # Calculate the angle of this vector in the standard coordinate system
    vector_angle <- atan2(vector_y, vector_x) * 180 / pi
    
    # Adjust the vector angle to match the orientation system (0 degrees is north, clockwise increase)
    adjusted_vector_angle <- (90 - vector_angle) %% 360
    
    # Calculate the absolute difference between the QB's orientation and the adjusted vector angle
    angle_diff <- abs(adjusted_vector_angle - qb_orientation) %% 360
    
    # Normalize the angle difference to be within the range [0, 180]
    angle_diff <- min(angle_diff, 360 - angle_diff)
    
    # Determine if the QB is oriented towards the receiver within the threshold
    return (if_else(angle_diff <= threshold,1,0))
  }

## Use NFL PBP to capture the target from each play
nfl_pbp<-as_tibble(load_pbp(seasons=2021)%>%
  select(play_id,
         game_id,
         old_game_id,
         home_team,
         away_team,
         week,
         season,
         play_type,
         desc,
         ydstogo,
         qtr,
         down,
         pass_length,
         pass_location,
         air_yards,
         yards_after_catch,
         yards_gained,
         qb_dropback,
         receiver,
         receiver_id,
         receiver_player_name,
         incomplete_pass)%>%
        mutate(gameId = as.integer(old_game_id),
               play_id = as.integer(play_id))%>%
        rename(playId = play_id))



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


### Import players and add identifiers

players<-read.csv("players.csv")
players<-add_player_identifiers(players,position_name="officialPosition")
plays<-read.csv('plays.csv')
games<-read.csv("games.csv")
pffData<-read.csv("pffScoutingData.csv")


full_df<-wk_std%>%
  left_join(pffData%>%select(nflId,gameId,playId,pff_role,pff_positionLinedUp),by=c("gameId","playId","nflId"))%>%
  left_join(plays,by=c("gameId","playId"))%>%
  left_join(nfl_pbp,by=c("gameId","playId"))%>%mutate(
    los = ifelse(playDirection == "left", 120 - absoluteYardlineNumber, absoluteYardlineNumber),
  )


### Need to Create Tracking Data Features for CB-WR interactions
## QB
qb_df <- full_df%>%
              filter(pff_role=='Pass')%>%
              select(gameId,
                          playId,
                          frameId,
                          nflId,
                          pff_role,
                          x,
                          y,
                          dir,
                          s,
                          o,
                          a)%>%
  rename_with(~ paste0("qb_", .), nflId:a)


## Ball
ball_df <- full_df%>%
  filter(is.na(pff_role))%>%
  select(gameId,
         playId,
         frameId,
         yardlineNumber,
         absoluteYardlineNumber,
         nflId,
         event,
         pff_role,
         x,
         y,
         dir,
         s,
         o,
         a)%>%
  rename_with(~ paste0("ball_", .), nflId:a)



## Receiver

receiver_df <- full_df%>%
  filter(pff_role=='Pass Route')


## Coverage

coverage_df <- full_df%>%
  filter(pff_role=='Coverage')%>%
  select(gameId,
         playId,
         frameId,
         nflId,
         pff_role,
         x,
         y,
         dir,
         s,
         o,
         a)%>%
  rename_with(~ paste0("coverage_", .), nflId:a)



### Join receiver, qb, and ball

feature_df<-receiver_df%>%
  left_join(qb_df,by=c('gameId','playId','frameId'))%>%
  left_join(ball_df,by=c('gameId','playId','frameId'))%>%
  mutate(dist_to_qb = euclidean_distance(x,y,qb_x,qb_y),
         dist_to_ball = euclidean_distance(x,y,ball_x,ball_y),
         qb_orient_rec = is_oriented_towards(qb_x,qb_y,qb_o,x,y))
         
### 3 closest cb

rec_xy<-full_df%>%
  filter(pff_role=='Pass Route')%>%
  select(gameId,
         playId,
         nflId,
         frameId,
         pff_role,
         x,
         y,
         dir,
         s,
         o,
         a)


cross_join_df <- rec_xy %>%
  inner_join(coverage_df, by = c("gameId", "playId","frameId"))

# Calculate the distances
cross_join_df <- cross_join_df %>%
  mutate(rec_cov_dist = euclidean_distance(x,y,coverage_x,coverage_y))

# Find the top 3 shortest distances for each gameId and playId
coverage_top_3 <- cross_join_df %>%
  group_by(gameId, playId,frameId,nflId) %>%
  arrange(rec_cov_dist) %>%
  slice_head(n = 3) %>%
  mutate(player_dist_rank = 1:n())%>%
  ungroup()

coverage_top_3%>%head(40)%>%View()




df_wide <- coverage_top_3 %>%
  pivot_wider(
    names_from = player_dist_rank,
    names_glue = "{.value}_{player_dist_rank}",
    values_from = c(coverage_nflId,coverage_pff_role, coverage_x, coverage_y,coverage_s,coverage_a,coverage_o,coverage_dir,rec_cov_dist)
  )


model_feat_df<-df_wide%>%
  inner_join(ball_df,by=c('gameId','playId','frameId'))%>%
  inner_join(qb_df,by=c('gameId','playId','frameId'))

### START HERE. ENGINEER MORE FEATURES. ADD DOWN DISTANCE ETC

play_feats<-full_df%>%select(
  gameId,
  playId,
  down.x,
  yardsToGo,
  defendersInBox,
  absoluteYardlineNumber,
  los,
  pff_passCoverageType,
  incomplete_pass,
  week.y
)%>%rename(down = down.x,
           week = week.y)%>%distinct()


#### Identify target on the play
rosters<-load_rosters(seasons=2021)

nfl_full_pbp<-load_pbp(seasons=2021)%>%select(play_id,
                                              game_id,
                                              old_game_id,
                                              receiver_player_name,
                                              receiver_player_id)%>%
  mutate(old_game_id = as.integer(old_game_id))%>%
  inner_join(rosters,by=c('receiver_player_id'='gsis_id'))%>%
  select(play_id,old_game_id,receiver_player_name,receiver_player_id,gsis_it_id)%>%mutate(gsis_it_id = as.integer(gsis_it_id))



target_data<-rec_xy%>%left_join(nfl_full_pbp,by=c("gameId"="old_game_id","playId"="play_id","nflId"="gsis_it_id"))

target_data%>%filter(gameId == 2021090900 & playId == 137)%>%View()


target_data <- target_data%>%
  mutate(is_target = ifelse(receiver_player_name == 'NA',0,1))%>%
  mutate(is_target = coalesce(is_target,0))%>%
  select(playId,nflId,frameId,gameId,is_target)




full_feat_df<-model_feat_df%>%
  left_join(target_data,by=c('gameId','playId','frameId','nflId'))

# Function to calculate distance to nearest sideline
calculate_sideline_distance <- function(y_position, sideline1 = 0, sideline2 = 53.3) {
  distance_to_sideline1 <- abs(y_position - sideline1)
  distance_to_sideline2 <- abs(y_position - sideline2)
  return(min(distance_to_sideline1, distance_to_sideline2))
}


### Need to create additional distance features, 
full_feat_df<-full_feat_df%>%left_join(play_feats%>%select(-absoluteYardlineNumber),by=c('gameId','playId'))


full_feat_df<-full_feat_df%>%mutate(
  dist_from_los = euclidean_distance(x,y,los,y),
  dist_from_qb = euclidean_distance(x,y,qb_x,qb_y),
  dist_from_ball = euclidean_distance(x,y,ball_x,ball_y),
  dist_from_side = sapply(y, calculate_sideline_distance)
)

library(caret)

# Assuming df is your data frame and pff_passCoverageType is the column to be encoded
dummies_model <- dummyVars(~ pff_passCoverageType, data = full_feat_df)
one_hot_encoded_df <- predict(dummies_model, newdata = full_feat_df)

# Convert the result to a data frame
one_hot_encoded_df <- as.data.frame(one_hot_encoded_df)

# Optionally, combine with the original data frame
df_combined <- cbind(full_feat_df, one_hot_encoded_df)

write_parquet(df_combined, paste0('target_comp_model_df','.parquet'))
