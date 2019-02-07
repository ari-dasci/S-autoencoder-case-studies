# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

## Data visualization
## Reference: Hinton, G.E., Salakhutdinov, R.R.: Reducing the dimensionality of data with neural networks

#' @import purrr
#' @import ruta
visualization <- function() {
  set.seed(12345)
  dataset <- read.csv("https://www.openml.org/data/get_csv/3619/dataset_186_satimage.arff")
  x_train <- dataset[, -ncol(dataset)] %>% as.matrix()

  codes <- lapply(2:3, function(code_length) {
    network <-
      input() +
      dense(12, activation = "relu") +
      dense(code_length, activation = "sigmoid") +
      dense(12, activation = "relu") +
      output("linear")

    model <- autoencoder_sparse(network) %>%
      train(x_train, epochs = 40, batch_size = 128)

    model %>% encode(x_train)
  })

  colors <- colorspace::heat_hcl(7) %>% colorspace::desaturate()
  symbols <- 1:7
  postscript("visualization_sat3.eps", width = 8, height = 8, paper = "special", horizontal = F)
  scatterplot3d::scatterplot3d(
    codes[[2]],
    color = colors[dataset$class],
    pch = symbols[dataset$class],
    xlab = "X",
    ylab = "Y",
    zlab = "Z"
  )
  legend(
    x = 6.2,
    y = 11,
    legend = c(
      "red soil",
      "cotton crop",
      "grey soil",
      "damp grey soil",
      "soil with vegetation",
      "mixture",
      "very damp grey soil"
    ),
    pch = symbols,
    col = colors,
    bg = "#ffffff",
    cex = 0.9
  )
  dev.off()
  postscript("visualization_sat2.eps", width = 8, height = 8, paper = "special", horizontal = F)
  plot(
    codes[[1]],
    col = colors[dataset$class],
    pch = symbols[dataset$class],
    xlab = "X",
    ylab = "Y"
  )
  legend(
    x = 0.7,
    y = 0.95,
    legend = c(
      "red soil",
      "cotton crop",
      "grey soil",
      "damp grey soil",
      "soil with vegetation",
      "mixture",
      "very damp grey soil"
    ),
    pch = symbols,
    col = colors,
    bg = "#ffffff",
    cex = .9
  )
  dev.off()
}
