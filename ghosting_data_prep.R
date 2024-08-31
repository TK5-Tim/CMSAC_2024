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

