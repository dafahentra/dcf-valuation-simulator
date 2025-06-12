#### Valuation ###
library(dplyr)
library(tidyverse)
library(ggplot2)
library(shiny)
library(shinydashboard)
library(plotly)

set.seed(10) # To get consistent random numbers
srevenue <- vector("numeric",0)
smargin <- rnorm(5,0.15,0.03) # Random generation of Net Operating Profit after Tax (NOPAT) margin
sgrowth <- rnorm(5,0.05,0.01) # Random generation of growth rates in revenue
sterminal_g <- rnorm(1,0.03,0.01) # Random generation of terminal growth rate
srevenue_0 <- 100
srevenue <- srevenue_0*cumprod(1+sgrowth)
snopat <- srevenue*smargin
sassets_to <- 1.3
s_assets <- srevenue/sassets_to # Using assets turnover ratio
sroic_5 <- smargin[5]*sassets_to # Using return on invested Capital ratio
sinv_rate_5 <- sterminal_g/sroic_5
sassets_0 <- 80
snet_inv <- vector("numeric",0)
snet_inv[1] <- s_assets[1]-sassets_0
snet_inv[2:5] <- diff(s_assets)
snet_inv[5] <- snopat[5]*sinv_rate_5
sfcff <- snopat-snet_inv
swacc <- 0.12
sdisc_factors <- 1/(1+swacc)^(1:5)
sterminal_value <- (sfcff[5]*(1+sterminal_g))/(swacc-sterminal_g)
sfcff[5] <- sfcff[5]+sterminal_value
sent_value <- sum(sfcff*sdisc_factors)

## Automating the function ###
s_ent_function <- function(srevenue_0=100,sassets_0=80,swacc=0.12,sassets_to=1.3,sgrowth=0.5,sgrowthstd=0.1,smargin=0.15,
                           smarginstd=0.03,sterminal_g=0.03,sterminalstd=0.01){
  library(dplyr)
  library(tidyverse)
  srevenue <- vector("numeric",0)
  smargin <- rnorm(5,smargin,smarginstd)
  sgrowth <- rnorm(5,sgrowth,sgrowthstd)
  sterminal_g <- rnorm(1,sterminal_g,sterminalstd)
  srevenue_0 <- 100
  srevenue <- srevenue_0*cumprod(1+sgrowth)
  snopat <- srevenue*smargin
  sassets_to <- 1.3
  s_assets <- srevenue/sassets_to
  sroic_5 <- smargin[5]*sassets_to
  sinv_rate_5 <- sterminal_g/sroic_5
  sassets_0 <- 80
  snet_inv <- vector("numeric",0)
  snet_inv[1] <- s_assets[1]-sassets_0
  snet_inv[2:5] <- diff(s_assets)
  snet_inv[5] <- snopat[5]*sinv_rate_5
  sfcff <- snopat-snet_inv
  swacc <- 0.12
  sdisc_factors <- 1/(1+swacc)^(1:5)
  sterminal_value <- (sfcff[5]*(1+sterminal_g))/(swacc-sterminal_g)
  sfcff[5] <- sfcff[5]+sterminal_value
  sent_value <- sum(sfcff*sdisc_factors)
  return(sent_value)
}

### Sensitivity Analysis - Exploring how WACC and growth rate impact the valuation ###

# Creating parameter grid
wacc_range <- seq(0.08, 0.16, by = 0.01)
growth_range <- seq(0.03, 0.07, by = 0.01)

# Matrix to store results of the combined possibilities
sensitivity_results <- expand.grid(WACC = wacc_range, Growth = growth_range)
sensitivity_results$Valuation <- mapply(function(wacc, growth) {
  s_ent_function(swacc = wacc, sgrowth = growth)
}, sensitivity_results$WACC, sensitivity_results$Growth)

# Plotting Sensitivity Analysis results in a heat map (Valuation vs. WACC and Growth)
ggplot(sensitivity_results, aes(x = WACC, y = Growth)) +
  geom_tile(aes(fill = Valuation)) +
  scale_fill_gradient(low = "green", high = "blue") +
  labs(title = "Sensitivity Analysis of Valuation", x = "WACC", y = "Growth Rate") +
  theme_minimal()

### Running Monte Carlo Simulations showing density of each range ###
svalues <- vector("numeric", 10^5)
for (i in 1:10^5) {
  svalues[i] <- s_ent_function()
}

plot_ly(x = svalues, type = "histogram", histnorm = "probability density", 
        marker = list(color = "skyblue", opacity = 0.7)) %>%
  layout(title = "Interactive Histogram of Monte Carlo Simulation",
         xaxis = list(title = "Valuation Value"),
         yaxis = list(title = "Density"))



### Using Shiny dashboard to create a mini-app to give detailed information ###
ui <- dashboardPage(
  dashboardHeader(title = "Business Valuation"),
  dashboardSidebar(
    sliderInput("wacc", "WACC:", min = 0.08, max = 0.16, value = 0.12, step = 0.01),
    sliderInput("growth", "Growth Rate:", min = 0.03, max = 0.1, value = 0.05, step = 0.01),
    sliderInput("margin", "Margin:", min = 0.10, max = 0.20, value = 0.15, step = 0.01),
    sliderInput("terminal_growth", "Terminal Growth Rate:", min = 0.02, max = 0.06, value = 0.03, step = 0.01),
    actionButton("run_simulation", "Run Simulation")
  ),
  dashboardBody(
    fluidRow(
      box(plotOutput("density_plot"), width = 12),
      box(tableOutput("summary_table"), width = 6)
    )
  )
)

# Define server logic
server <- function(input, output) {
  
  observeEvent(input$run_simulation, {
    # Run the Monte Carlo simulations
    svalues <- vector("numeric", 10^5)
    for (i in 1:10^5) {
      svalues[i] <- s_ent_function(
        swacc = input$wacc, 
        sgrowth = input$growth, 
        smargin = input$margin,
        sterminal_g = input$terminal_growth
      )
    }
    
    # To generate the density plot
    output$density_plot <- renderPlot({
      ggplot(data.frame(value = svalues), aes(x = value)) +
        geom_density(fill = "skyblue", alpha = 0.7) +
        labs(title = "Monte Carlo Simulation for Business Valuation",
             x = "Valuation Value",
             y = "Density") +
        theme_minimal()
    })
    
    #  To Calculate summary statistics
    summary_stats <- summary(svalues)
    summary_table <- data.frame(
      Statistic = c("Min", "1st Quartile", "Median", "Mean", "3rd Quartile", "Max"),
      Value = as.numeric(c(summary_stats[1], summary_stats[2], summary_stats[3], summary_stats[4], summary_stats[5], summary_stats[6]))
    )
    
    # To give the summary of the values in the dashboard in the form of a table
    output$summary_table <- renderTable({
      summary_table
    })
  })
}

# Run the application
shinyApp(ui = ui, server = server)