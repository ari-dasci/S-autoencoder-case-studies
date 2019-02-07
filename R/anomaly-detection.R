# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

## Anomaly detection
## Reference: Sakurada, Yairi. Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction

#' @import purrr
#' @import ruta
anomaly_detection <- function() {
  set.seed(12345)
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

  network1 <-
    input() +
    dense(16, activation = "sigmoid") +
    output()

  loss <- "mean_squared_error"

  model <- autoencoder_denoising(network1, loss, noise_type = "saltpepper", p = 0.1) %>%
    train(x_train, epochs = 50, batch_size = 32)
  reconstructions <- model %>% reconstruct(x_test)
  errors <- rowMeans((reconstructions - x_test) ** 2)
  postscript("anomaly_detection_lorenz.eps", width = 8, height = 8, paper = "special", horizontal = F)
  plot(
    x = ts$time[in_test],
    y = errors1,
    #col = 1 + is_anomaly[in_test],
    col = c("#909090", "#000000")[1 + (errors1 > mean(errors1) + sd(errors1))],
    type = "h",
    xlab = "Time (s)",
    ylab = "Reconstruction error"
  )
  lines(x = rep(ts$time[is_anomaly][1], 2), y = c(0, 5), lty = 2, col = "#aaaaaa")
  lines(x = rep(ts$time[is_anomaly][length(ts$time[is_anomaly])], 2), y = c(0, 5), lty = 2, col = "#aaaaaa")
  dev.off()
}
