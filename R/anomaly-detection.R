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
anomaly_detection <- function() {
  set.seed(12345)

  dat <- load_unsw()

  network1 <-
    input() +
    dense(10, activation = "relu") +
    output("sigmoid")

  # loss <- "mean_squared_error"
  loss <- "binary_crossentropy"

  model <- autoencoder_denoising(network1, loss, noise_type = "saltpepper", p = 0.1)
  # model <- autoencoder_contractive(network1, loss)
  model <- ruta::train(model, dat$train$x, epochs = 10, batch_size = 32)
  reconstructions <- model %>% reconstruct(dat$test$x)
  errors <- rowMeans((reconstructions - dat$test$x) ** 2)
  # postscript("anomaly_detection_lorenz.eps", width = 8, height = 8, paper = "special", horizontal = F)
  plot(
    x = dat$test$t,
    y = errors,
    # Mark in black normal instances, in red undetected attacks or false alarms, in green true alarms
    col = 2 + dat$test$y - !(errors > mean(errors, na.rm = T) + sd(errors, na.rm = T)),
    # col = c("#909090", "#000000")[1 + (errors > mean(errors, na.rm = T) + sd(errors, na.rm = T))],
    type = "h",
    xlab = "Time (s)",
    ylab = "Reconstruction error"
  )
  # lines(x = rep(dat$test$t[dat$test$y][1], 2), y = c(0, 5), lty = 2, col = "#aaaaaa")
  # lines(x = rep(dat$test$t[dat$test$y][length(dat$test$t[dat$test$y])], 2), y = c(0, 5), lty = 2, col = "#aaaaaa")
  # dev.off()
}
