library(tidymodels)
library(forecast)
library(vroom)
library(ggplot2)
library(patchwork)
library(kknn)
library(modeltime)
library(timetk)

train <- vroom('train.csv')
test <- vroom('test.csv')

train_1 <- train |> filter(item == 1, store == 2)
train_2 <- train |> filter(item == 3, store == 4)

test_1 <- test |> filter(item == 1, store == 2)
test_2 <- test |> filter(item == 1, store == 2)

## EDA
sub1 <- train |>
  filter(store == 1, item == 1)
sub2 <- train |>
  filter(store == 2, item == 2)

ts1 <- sub1 |>
  ggplot(mapping = aes(x = date, y = sales)) +
  geom_line() + 
  geom_smooth(se = FALSE)

ts2 <- sub2 |>
  ggplot(mapping = aes(x = date, y = sales)) +
  geom_line() + 
  geom_smooth(se = FALSE)

ac_month1 <- sub1 |>
  pull(sales) |>
  forecast::ggAcf(lag.max = 30)

ac_month2 <- sub2 |>
  pull(sales) |>
  forecast::ggAcf(lag.max = 30)

ac_year1 <- sub1 |>
  pull(sales) |>
  forecast::ggAcf(lag.max = 730)

ac_year2 <- sub2 |>
  pull(sales) |>
  forecast::ggAcf(lag.max = 730)

(ts1 + ts2) / (ac_month1 + ac_month2) / (ac_year1 + ac_year2)

## Feature Engineering
train <- train |>
  filter(store == 1, item == 2)

test <- test |>
  filter(store == 1, item == 2)

my_recipe <- recipe(sales ~ ., data = train) |>
  step_date(date, features = 'dow') |>
  step_date(date, features = 'decimal') |>
  step_date(date, features = 'month')

knn_mod <- nearest_neighbor(neighbors = tune()) |>
  set_mode('regression') |>
  set_engine('kknn')

knn_wf <- workflow () |>
  add_recipe(my_recipe) |>
  add_model(knn_mod) 

tuning_grid <- grid_regular(neighbors(),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats = 1)

CV_results <- knn_wf |>
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(smape))

res <- CV_results |> show_best(metric = 'smape', n = 1)

## ARIMA
cv_split <- time_series_split(train_1, assess = '3 months', cumulative = TRUE)
tscv1 <- cv_split |>
  tk_time_series_cv_plan() |>
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

arima_rec <- recipe(sales ~ date, data = train_1)

arima_mod <- arima_reg(seasonal_period = 365,
                       non_seasonal_ar = 5,
                       non_seasonal_ma = 5,
                       seasonal_ar = 2,
                       seasonal_ma = 2,
                       non_seasonal_differences = 2,
                       seasonal_differences = 2) |>
  set_engine('auto_arima')

arima_wf <- workflow() |>
  add_recipe(arima_rec) |>
  add_model(arima_mod) |>
  fit(data = training(cv_split))

cv_results <- modeltime_calibrate(arima_wf, 
                                   new_data = testing(cv_split))

cv_results |> modeltime_forecast(
  new_data = testing(cv_split),
  actual_data = train_1) |>
    plot_modeltime_forecast(.interactive = FALSE)
  
fullfit <- cv_results |>
  modeltime_refit(data = train_1)

fore1 <- fullfit |>
  modeltime_forecast(
    new_data = test_1,
    actual_data = train_1
  ) |>
  plot_modeltime_forecast(.interactive = FALSE)


cv_split2 <- time_series_split(train_2, assess = '3 months', cumulative = TRUE)
tscv2 <- cv_split2 |>
  tk_time_series_cv_plan() |>
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

arima_rec2 <- recipe(sales ~ date, data = train_2)

arima_mod2 <- arima_reg(seasonal_period = 365,
                       non_seasonal_ar = 5,
                       non_seasonal_ma = 5,
                       seasonal_ar = 2,
                       seasonal_ma = 2,
                       non_seasonal_differences = 2,
                       seasonal_differences = 2) |>
  set_engine('auto_arima')

arima_wf2 <- workflow() |>
  add_recipe(arima_rec2) |>
  add_model(arima_mod2) |>
  fit(data = training(cv_split2))

cv_results2 <- modeltime_calibrate(arima_wf2, 
                                  new_data = testing(cv_split2))

cv_results2 |> modeltime_forecast(
  new_data = testing(cv_split2),
  actual_data = train_2) |>
  plot_modeltime_forecast(.interactive = FALSE)

fullfit2 <- cv_results2 |>
  modeltime_refit(data = train_2)

fore2 <- fullfit2 |>
  modeltime_forecast(
    new_data = test_2,
    actual_data = train_2
  ) |>
  plot_modeltime_forecast(.interactive = FALSE)

(tscv1 + tscv2) / (fore1 + fore2)

## Facebook Prophet Model
prophet_model <- prophet_reg() |>
  set_engine(engine = 'prophet') |>
  fit(sales ~ date, data = training(cv_split))

cv_results_fb <- modeltime_calibrate(prophet_model, 
                                  new_data = testing(cv_split))

cv1 <- cv_results_fb |> modeltime_forecast(
  new_data = testing(cv_split),
  actual_data = train_1) |>
  plot_modeltime_forecast(.interactive = FALSE)

fullfit_fb <- cv_results_fb |>
  modeltime_refit(data = train_1)

fore1_fb <- fullfit_fb |>
  modeltime_forecast(
    new_data = test_1,
    actual_data = train_1
  ) |>
  plot_modeltime_forecast(.interactive = FALSE)


prophet_model2 <- prophet_reg() |>
  set_engine(engine = 'prophet') |>
  fit(sales ~ date, data = training(cv_split2))

cv_results_fb2 <- modeltime_calibrate(prophet_model2, 
                                     new_data = testing(cv_split2))

cv2 <- cv_results_fb2 |> modeltime_forecast(
  new_data = testing(cv_split2),
  actual_data = train_2) |>
  plot_modeltime_forecast(.interactive = FALSE)

fullfit_fb2 <- cv_results_fb2 |>
  modeltime_refit(data = train_2)

fore2_fb <- fullfit_fb2 |>
  modeltime_forecast(
    new_data = test_2,
    actual_data = train_2
  ) |>
  plot_modeltime_forecast(.interactive = FALSE)

(cv1 + cv2) / (fore1_fb + fore2_fb)


