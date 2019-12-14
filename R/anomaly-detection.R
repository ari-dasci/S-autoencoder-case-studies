# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

## Anomaly detection
## Reference: Sakurada, Yairi. Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction

load_unsw <- function() {
  train_name <- "UNSW_NB15_training-set.csv"
  test_name <- "UNSW_NB15_testing-set.csv"
  if (!file.exists(train_name)) {
    download.file("https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW_NB15_training-set.csv", train_name)
  }
  if (!file.exists(test_name)) {
    download.file("https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW_NB15_testing-set.csv", test_name)
  }
  x_train <- read.csv(train_name)
  # use only Normal instances for training
  x_train <- x_train[x_train$label == 0, 2:43] # remove id, attack and label columns
  x_test <- read.csv(test_name)
  y_test <- x_test$label
  x_test <- x_test[, 2:43]

  nominal <- which(sapply(x_train, is.factor))

  for (v in nominal) {
    x_test[, v] <- factor(x_test[, v], levels = levels(x_train[, v]))
    # levels(x_test[, v]) <- levels(x_train[, v])
  }

  dummy <- caret::dummyVars(~ ., data = x_train[, nominal], fullRank=T)
  extravars <- predict(dummy, newdata = x_train[, nominal])
  x_train <- cbind(x_train[, -nominal], extravars)
  scale_vars <- list(mx = apply(x_train, 2, max), mn = apply(x_train, 2, min))
  diff <- sapply(scale_vars$mx - scale_vars$mn, function(x) max(x, 1))
  for (i in 1:ncol(x_train)) {
    x_train[, i] <- (x_train[, i] - scale_vars$mn[i]) / diff[i]
  }

  # dummy <- caret::dummyVars(~ ., data = x_test[, nominal], fullRank=T)
  extravars <- predict(dummy, newdata = x_test[, nominal])
  x_test <- cbind(x_test[, -nominal], extravars)
  for (i in 1:ncol(x_test)) {
    x_test[, i] <- (x_test[, i] - scale_vars$mn[i]) / diff[i]
  }

  list(
    train = list(x = data.matrix(x_train), t = 1:nrow(x_train)),
    test = list(x = data.matrix(x_test), y = y_test, t = 1:nrow(x_test))
  )
}

load_artificial <- function() {
  # Artificial dataset using Lorenz equations as described in Sakurada, Yairi.
  ts <- nonlinearTseries::lorenz(time = seq(0.1, 100, 0.1), do.plot = FALSE)
  which_anomaly <- 800:900
  is_anomaly <- 1:1000 %in% which_anomaly
  in_test <- 701:1000
  ts$x[which_anomaly] <- ts$x[sample(which_anomaly)]
  z <- rbind(ts$x, ts$y, ts$z)
  w <- array(runif(25 * 3, min = -5, max = 5), dim = c(25, 3))
  x <- w %*% z
  x_train <- t(x)[-in_test, ] %>% scale
  x_test <- t(x)[in_test, ] %>% scale(center = x_train %@% "scaled:center", scale = x_train %@% "scaled:scale")

  list(
    train = list(x = x_train, t = ts$time[-in_test]),
    test = list(x = x_test, y = is_anomaly[in_test], t = ts$time[in_test])
  )
}

#' @import purrr
#' @import ruta
#' @import keras
anomaly_detection <- function() {
  set.seed(12345)
  allow_growth()
  dat <- load_unsw()

  autoencoder <- keras_model_sequential(list(
    # layer_gaussian_noise(input_shape = 187L, stddev = .1),
    layer_dense(units = 2L, input_shape = 187L, activation = "relu"),
    # layer_dense(units = 2L, activation = "sigmoid"),
    # layer_dense(units = 50L, activation = "sigmoid"),
    layer_dense(units = 187L)
  ))

  compile(autoencoder, loss = "mean_squared_error", optimizer = "adam")
  fit(autoencoder, dat$train$x, dat$train$x, epochs = 5, batch_size = 256)

  reconstructions <- predict(autoencoder, dat$test$x)
  train_reconstructions <- predict(autoencoder, dat$train$x)

  train_errors <- rowMeans((train_reconstructions - dat$train$x) ** 2, na.rm = T)
  errors <- rowMeans((reconstructions - dat$test$x) ** 2, na.rm = T)

  results <- data.frame(t=dat$test$t, y=dat$test$y, error=errors)
  results <- results[!is.na(results$error),]

  detect_over <- function(times) {
    results$error > mean(train_errors, na.rm = T) + times * sd(train_errors, na.rm = T)
  }
  detection <- detect_over(6)
  results$type = ifelse(results$y == 0, ifelse(detection, "FP", "TN"), ifelse(detection, "TP", "FN"))

  library(ggplot2)
  postscript("anomaly_histogram.eps", width = 6, height = 6, paper = "special", horizontal = F)
  ggplot(results, aes(x=error, fill=type)) +
    geom_histogram(bins = 120) +
    scale_y_continuous("Count") + scale_x_log10("Reconstruction error") +
    theme(legend.position = "bottom") +
    scale_fill_viridis_d("Answer type", direction = -1)
  dev.off()
  postscript("anomaly_column.eps", width = 6, height = 6, paper = "special", horizontal = F)
  ggplot(results, aes(color=type)) +
    geom_segment(aes(x=t,xend=t,y=0,yend=error)) +
    scale_y_log10("Reconstruction error") +
    scale_x_discrete("Time of request") +
    theme(legend.position = "bottom") +
    scale_color_viridis_d("Answer type", direction = -1)
  dev.off()

  metrics_per_times <- compiler::cmpfun(function(start = 0, end = 100) {
    t(as.data.frame(lapply(seq(start, end, 1), function(i) {
      true <- dat$test$y == 1
      pred <- detect_over(i)
      tp = sum(true & pred, na.rm = T)
      fp = sum(!true & pred, na.rm = T)
      tn = sum(!true & !pred, na.rm = T)
      fn = sum(true & !pred, na.rm = T)
      return(c(
        times = i,
        tp = tp,
        fp = fp,
        tn = tn,
        fn = fn,
        precision = tp / (tp + fp),
        recall = tp / (tp + fn)
      ))
    }), fix.empty.names = F))
  })

  metrics_per_threshold <- function(start = 0, end = 100) {
    t(as.data.frame(lapply(seq(log(1e-6+start), log(1e-6+end), .1), function(i) {
      i <- exp(i)
      true <- dat$test$y == 1
      pred <- errors > i
      tp = sum(true & pred, na.rm = T)
      fp = sum(!true & pred, na.rm = T)
      tn = sum(!true & !pred, na.rm = T)
      fn = sum(true & !pred, na.rm = T)
      return(c(
        n = i,
        tp = tp,
        fp = fp,
        tn = tn,
        fn = fn,
        precision = tp / (tp + fp),
        recall = tp / (tp + fn)
      ))
    }), fix.empty.names = F))
  }

  result <- as.data.frame(metrics_per_threshold(end = 158))
  ggplot(result, aes(x=recall,y=precision)) + geom_line() + scale_y_continuous(limits=c(0, 1))

}
