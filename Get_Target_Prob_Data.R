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
  left_join(nfl_pbp,by=c("gameId","playId"))
