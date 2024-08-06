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
### Import tracking Data, standardize and convert to parquet files

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
nfl_pbp<-nflreadr::load_pbp(2023)

str(nfl_pbp)
str(games)
str(pffData)
str(players)
str(wk_std)


# Test the function with some sample data
#animate_play(gameid = 2021090900, playid = 97, track_df = wk_std)



pffData%>%filter(gameId==2021090900 & playId ==97)


pffData%>%select(pff_role)%>%unique()


### Need to createa a dataframe of pass rushers and generate features off the passer and pass blocker

### features
## time since snap
## down
## distance
## score diff
## distance to qb
## orientation to 3 closest blockers
## speed 
## qb speed
## outcome (pff_sack + pff_hit + pff_huryy = pressure)


pff_join_df<-pffData%>%select(gameId,
                 playId,
                 nflId,
                 pff_role,
                 pff_hit,
                 pff_hurry,
                 pff_sack,
                 pff_beatenByDefender,
                 pff_hurryAllowed,
                 pff_hitAllowed,
                 pff_sackAllowed,
                 pff_nflIdBlockedPlayer,
                 pff_blockType)


pressure_df<-wk_std%>%
  left_join(pff_join_df ,by=c("gameId","playId","nflId"))%>%
  mutate_all(~ ifelse(is.na(.), 0, .))%>%
  mutate(is_pressure = ifelse(pff_hit + pff_hurry + pff_sack>0,1,0))


rusher_df<-pressure_df%>%
  filter(pff_role=='Pass Rush')%>%select(-pff_hurryAllowed,
                                         -pff_sackAllowed,
                                         -pff_hitAllowed,
                                         -pff_beatenByDefender,
                                         -pff_blockType,
                                         -pff_nflIdBlockedPlayer)

rush_join_df<-rusher_df%>%select(gameId,playId,frameId,nflId,x,y)

qb_df<- pressure_df%>%
  filter(pff_role=='Pass')%>%
  select(gameId,playId,nflId,frameId,x,y,s,a,dis,o,dir)%>%
  rename_with(~ paste0("qb_", .), x:dir)%>%
  rename(qb_nflId = nflId)

blocker_df<-pressure_df%>%
  filter(pff_role=='Pass Block')%>%
  rename_with(~ paste0("blocker_", .), x:dir)%>%
  select(-pff_hit,pff_hurry,pff_sack,time)

blk_join_df<-blocker_df%>%
  select(gameId,
         playId,
         frameId,
         nflId,
         blocker_x,
         blocker_y,
         blocker_s,
         blocker_a,
         blocker_o,
         blocker_dir,
         blocker_dis,
         pff_role,
         pff_beatenByDefender,
         pff_hurryAllowed,
         pff_hitAllowed,
         pff_sackAllowed,
         pff_blockType)%>%
  rename(blocker_id = nflId,
         blocker_role = pff_role)

### Distance Function Calculation
euclidean_distance <- function(x1, y1, x2, y2) {
  sqrt((x1 - x2)^2 + (y1 - y2)^2)}

cross_join_df <- rush_join_df %>%
  inner_join(blk_join_df, by = c("gameId", "playId","frameId"))

# Calculate the distances
cross_join_df <- cross_join_df %>%
  mutate(blk_rush_dist = euclidean_distance(x,y,blocker_x,blocker_y))

# Find the top 3 shortest distances for each gameId and playId
blocker_top_3 <- cross_join_df %>%
  group_by(gameId, playId,frameId,nflId) %>%
  arrange(blk_rush_dist) %>%
  slice_head(n = 3) %>%
  mutate(player_dist_rank = 1:n())%>%
  ungroup()

blocker_top_3%>%head(40)%>%View()
  
  
pres_long_df<-blocker_top_3%>%
  inner_join(qb_df,by=c("gameId","playId","frameId"))%>%
  inner_join(rusher_df,by=c("gameId","playId","frameId","nflId","x","y"))


####### START HERE. NEED TO CREATE SOME ANGLE VARS AND OTHER IMPORTANT FEATURES BETWEEN PLAYERS


# Function to calculate the angle between two vectors
calculate_angle <- function(x1, y1, x2, y2) {
  dot_product <- x1 * x2 + y1 * y2
  magnitude1 <- sqrt(x1^2 + y1^2)
  magnitude2 <- sqrt(x2^2 + y2^2)
  cos_angle <- dot_product / (magnitude1 * magnitude2)
  angle <- acos(pmin(pmax(cos_angle, -1), 1)) * (180 / pi)  # Convert to degrees and clamp value between -1 and 1 to avoid NaN
  return(angle)
}

# Function to calculate cosine similarity between two vectors
cosine_similarity <- function(x1, y1, x2, y2) {
  dot_product <- x1 * x2 + y1 * y2
  magnitude1 <- sqrt(x1^2 + y1^2)
  magnitude2 <- sqrt(x2^2 + y2^2)
  cos_sim <- dot_product / (magnitude1 * magnitude2)
  return(cos_sim)
}

# Function to calculate orthogonal distance from point to line
orthogonal_distance <- function(x1, y1, x2, y2, x3, y3) {
  num <- abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1)
  den <- sqrt((y2 - y1)^2 + (x2 - x1)^2)
  return(num / den)
}


# Function to calculate distance to nearest sideline
calculate_sideline_distance <- function(y_position, sideline1 = 0, sideline2 = 53.3) {
  distance_to_sideline1 <- abs(y_position - sideline1)
  distance_to_sideline2 <- abs(y_position - sideline2)
  return(min(distance_to_sideline1, distance_to_sideline2))
}

# Calculate leverage angles for every frame
df_with_angles <- pres_long_df%>%
  mutate(
    # Vectors from rusher to QB
    vec_rusher_to_qb_x = qb_x - x,
    vec_rusher_to_qb_y = qb_y - y,
    
    # Vectors from rusher to blocker
    vec_rusher_to_blocker_x = blocker_x - x,
    vec_rusher_to_blocker_y = blocker_y - y,
    
    # Calculate leverage angles
    leverage_angle = calculate_angle(vec_rusher_to_qb_x, vec_rusher_to_qb_y, vec_rusher_to_blocker_x, vec_rusher_to_blocker_y),
    
    # Cosine similarity between rusher-QB and rusher-blocker vectors
    cos_sim = cosine_similarity(vec_rusher_to_qb_x, vec_rusher_to_qb_y, vec_rusher_to_blocker_x, vec_rusher_to_blocker_y),
    
    # Orthogonal distance from blocker to rusher-QB vector
    ortho_dist = orthogonal_distance(x, y, qb_x, qb_y, blocker_x, blocker_y)
  )%>%
mutate(blocker_influence = if_else(ortho_dist <= 1, 1, exp(-ortho_dist)),
      rel_s = s - qb_s,
      approach_angle = atan2(y - qb_y, x - qb_x),
      rel_o = abs(o - qb_o),
      qb_dist_near_sideline = sapply(qb_y, calculate_sideline_distance),
      rush_qb_dist = euclidean_distance(x,y,qb_x,qb_y))

# Summarize the blocker-interference for each rusher
blocker_interference_df <- df_with_angles %>%
  group_by(gameId, playId, frameId, nflId) %>%
  summarise(blocker_interference = sum(blocker_influence, na.rm = TRUE), .groups = 'drop')

# Display the resulting dataframe
print(blocker_interference_df)


df_wide <- df_with_angles %>%
  pivot_wider(
    names_from = player_dist_rank,
    names_glue = "{.value}_{player_dist_rank}",
    values_from = c(blocker_id, blocker_x, blocker_y,blocker_s,blocker_a,blocker_o,blocker_dir,blocker_dis,pff_blockType,pff_beatenByDefender,pff_hurryAllowed,pff_hitAllowed,pff_sackAllowed,blk_rush_dist,vec_rusher_to_blocker_x,vec_rusher_to_blocker_y,leverage_angle,cos_sim,ortho_dist,blocker_influence)
  )


### START HERE. ENGINEER MORE FEATURES. ADD DOWN DISTANCE ETC

play_feats<-plays%>%select(
  gameId,
  playId,
  down,
  yardsToGo,
  defendersInBox,
  absoluteYardlineNumber
)

pres_model_df<-df_wide%>%
  inner_join(play_feats,by=c('gameId','playId'))

###

write_parquet(pres_model_df, paste0('pressure_model_df','.parquet'))






  
  