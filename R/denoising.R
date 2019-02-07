# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

## Image denoising

#' @import purrr
denoising <- function() {
  set.seed(12345)
  dataset <- keras::dataset_cifar10()
  x_train <- (dataset$train$x ) / 255

  network <- input() +
    conv(16, 3, upsampling = 2, activation = "relu") +
    conv(16, 3, max_pooling = 2, activation = "relu") +
    conv(3, 3, activation = "sigmoid")

  if (file.exists("denoising.tar.gz")) {
    model <- load_from("denoising.tar.gz")
  } else {
    model <-
      autoencoder_denoising(network, loss = "binary_crossentropy", noise_type = "gaussian", sd = .05) %>%
      train(x_train, epochs = 30, batch_size = 500, optimizer = "adam")

    model %>% save_as("denoising.tar.gz", dir = ".")
  }

  x_test <- (dataset$test$x) / 255
  noisy <- noise_gaussian(sd = .05) %>% apply_filter(x_test)
  recovered <- model %>% reconstruct(noisy)

  x_test <- x_test %>% keras::array_reshape(c(dim(x_test)[1], prod(dim(x_test)[-1])))
  noisy <- noisy %>% keras::array_reshape(dim(x_test))
  recovered <- recovered %>% keras::array_reshape(dim(x_test))
  selected_idx <- sample(dim(x_test)[1], 10)[-c(2,5,8)]
  pdf("denoising_cifar10.pdf", width = 7, height = 3)
  plot_sample(x_test[selected_idx, ], noisy[selected_idx, ], recovered[selected_idx, ])
  dev.off()
}
