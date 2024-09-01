library(arrow)
library(ggplot2)
library(dplyr)

players<-read.csv("players.csv")
#players<-add_player_identifiers(players,position_name="officialPosition")
plays<-read.csv('plays.csv')
games<-read.csv("games.csv")
pffData<-read.csv("pffScoutingData.csv")

pres_df<-read_parquet('full_pressure_preds.parquet')

pres_df<-pres_df%>%
  left_join(players%>%select(nflId,officialPosition,displayName),by='nflId')

pres_df%>%select(frameId,
                 is_pressure,
                 pressure_prob)%>%
  group_by(frameId,
           is_pressure)%>%
  mutate(average_prob = mean(pressure_prob))%>%
  select(-pressure_prob)%>%
  arrange(frameId,is_pressure)%>%ungroup()%>%distinct()%>%
  View()


cor(pres_df$pressure_prob,pres_df$rush_qb_dist)


# Assuming pres_df is your data frame
# Round pressure_prob to the first decimal
pres_df <- pres_df %>%
  mutate(pressure_prob_rounded = round(pressure_prob, 1))

# Group by the rounded pressure_prob and calculate summary statistics for rush_qb_dist
summary_stats <- pres_df %>%
  group_by(pressure_prob_rounded) %>%
  summarize(
    count = n(),
    mean_rush_qb_dist = mean(rush_qb_dist, na.rm = TRUE),
    sd_rush_qb_dist = sd(rush_qb_dist, na.rm = TRUE),
    min_rush_qb_dist = min(rush_qb_dist, na.rm = TRUE),
    max_rush_qb_dist = max(rush_qb_dist, na.rm = TRUE)
  )

# Print the summary statistics
print(summary_stats)

# Calculate the correlation between pressure_prob and rush_qb_dist
correlation <- cor(pres_df$pressure_prob, pres_df$rush_qb_dist, use = "complete.obs")

# Print the correlation
print(correlation)



# Install and load required packages
if (!require(ggridges)) {
  install.packages("ggridges")
}
library(ggridges)
library(ggplot2)
library(dplyr)

# Assuming pres_df is your data frame
# Round pressure_prob to the first decimal
pres_df <- pres_df %>%
  mutate(pressure_prob_rounded = round(pressure_prob, 1))

# Calculate the mean rush_qb_dist for each rounded pressure_prob
mean_df <- pres_df %>%
  group_by(pressure_prob_rounded) %>%
  summarize(mean_rush_qb_dist = mean(rush_qb_dist, na.rm = TRUE))

# Create the ridge plot with means
ggplot(pres_df, aes(x = rush_qb_dist, y = factor(pressure_prob_rounded), fill = factor(pressure_prob_rounded))) +
  geom_density_ridges(scale = 0.9, alpha = 0.7) +
  geom_point(data = mean_df, aes(x = mean_rush_qb_dist, y = factor(pressure_prob_rounded)), color = "red", size = 3) +
  labs(
    title = "Distribution of Rush QB Distance by Rounded Pressure Probability",
    x = "Rush QB Distance",
    y = "Rounded Pressure Probability"
  ) +
  theme_ridges() +
  theme(legend.position = "none")


avg_pres<-pres_df%>%select(frameId,
                 officialPosition,
                 is_pressure,
                 pressure_prob)%>%
  group_by(frameId,
           officialPosition,
           is_pressure)%>%
  mutate(average_prob = mean(pressure_prob),
         median_prob = median(pressure_prob))%>%
  select(-pressure_prob)%>%
  arrange(frameId,officialPosition,is_pressure)%>%ungroup()%>%distinct()

ggplot(avg_pres,aes(x=frameId,y=average_prob,groups=is_pressure))+geom_point()
ggplot(avg_pres,aes(x=frameId,y=median_prob,groups=is_pressure))+geom_point()

ggplot(avg_pres,aes(x=frameId,y=average_prob,groups=is_pressure))+geom_point()

### look at pressure probability by position group

### number of predictions by frameId. cap at frame 50? 

ggplot(data=pres_df%>%sample_n(50000),aes(x=rush_qb_dist,y=pressure_prob))+geom_point()