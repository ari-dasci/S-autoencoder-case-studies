# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

## Data visualization
## Reference: Hinton, G.E., Salakhutdinov, R.R.: Reducing the dimensionality of data with neural networks

#' @import purrr
#' @import ruta
visualization <- function(x, y, classnames=NULL, epochs = 100, batch_size = 8) {
  set.seed(12345)

  in_test <- caret::createDataPartition(y, p=0.2)[[1]]

  code_length <- 2
  # We aim for the same ratio of reduction in each layer of the encoder
  middle <- round(sqrt(code_length * ncol(x)))

  network <-
    input() +
    dense(middle, activation = "relu") +
    dense(code_length, activation = "sigmoid") +
    dense(middle, activation = "relu") +
    output("linear")

  model_name <- paste0("model_vis_", if (is.null(classnames)) "reg" else length(classnames), ".tar.gz")
  model <- if (file.exists(model_name)) {
    ruta::load_from(model_name)
  } else {
    autoencoder_contractive(network, weight = 1e-04) %>%
      train(x[-in_test,], epochs = epochs, batch_size = batch_size, optimizer = "adam")
  }

  model %>% save_as(model_name, dir = ".")
  codes <- model %>% encode(x)
  codes <- as.data.frame(codes)
  colnames(codes) <- c("V1", "V2")

  # Comparison to PCA
  pca <- prcomp(x[-in_test,], rank. = 2)
  back_train <- t(t(pca$x %*% t(pca$rotation)) + pca$center)
  forward_test <- predict(pca, x[in_test,])
  back_test <- t(t(forward_test %*% t(pca$rotation)) + pca$center)
  mse_train <- ruta::evaluate_mean_squared_error(model, x[-in_test,])[[2]]
  mse_test <- ruta::evaluate_mean_squared_error(model, x[in_test,])[[2]]
  # cat("Train:", nrow(x)-length(in_test), "Test:", length(in_test))
  cat("PCA - Train:", mean((x[-in_test,] - back_train)**2), "Test:", mean((x[in_test,] - back_test)**2), "\n")
  cat("Autoencoder - Train:", mse_train, "Test:", mse_test, "\n")

  if (is.null(classnames)) {
    aesthetic <- ggplot2::aes(x=V1,y=V2,color= y)
    colors <- ggplot2::scale_color_viridis_c()
  } else {
    classes <- classnames[as.integer(y) + (if (min(y) == 0) 1 else 0)]
    aesthetic <- ggplot2::aes(x=V1,y=V2,color=classes, shape=classes)
    colors <- ggplot2::scale_color_viridis_d()
  }

  ggplot2::ggplot(codes, aesthetic) +
    ggplot2::geom_point(alpha=1/2) + ggplot2::theme(legend.position="bottom") + colors
}

visualization_satellite <- function() {
  dataset <- read.csv("https://www.openml.org/data/get_csv/3619/dataset_186_satimage.arff")
  x_train <- dataset[, -ncol(dataset)] %>% as.matrix()
  y <- dataset$class
  classnames <- c(
    "red soil",
    "cotton crop",
    "grey soil",
    "damp grey soil",
    "soil with vegetation",
    "mixture",
    "very damp grey soil"
  )
  pdf("visualization_sat2.pdf", width = 6, height = 6, paper = "special")
  visualization(x_train, y, classnames = classnames)
  dev.off()
}

visualization_cpu <- function() {
  data2 <- read.csv("https://www.openml.org/data/get_csv/52751/cpu_act.arff", stringsAsFactors = F)
  x_train <- data2[, -ncol(data2)] %>% as.matrix()
  x_train <- scale(x_train)
  y <- data2$usr
  pdf("visualization_cpu.pdf", width = 6, height = 6, paper = "special")
  visualization(x_train, y, epochs=30)
  dev.off()
}
