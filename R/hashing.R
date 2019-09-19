# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

## Hashing
## Reference: Salakhutdinov, Hinton. Semantic hashing

#' @import purrr
#' @import ruta
#' @importFrom mldr.datasets bibtex
hashing <- function() {
  set.seed(12345)
  n <- 1000

  # dataset <- keras::dataset_imdb(num_words = n)
  dataset <- bibtex()
  dataset <- dataset$dataset[, dataset$attributesIndexes]
  dataset <- apply(dataset, 2, as.numeric)

  in_test <- sample(1:nrow(dataset), floor(nrow(dataset)/5)) # 80-20 split

  x_train <- as.matrix(dataset[-in_test, ])
  x_test <- as.matrix(dataset[in_test, ])

  # x_train <- matrix(0, nrow = length(dataset$train$x), ncol = n)
  # x_train <- t(sapply(dataset$train$x, function(instance) {
  #   as.numeric(1:n %in% instance)
  # }))
  # x_test <- t(sapply(dataset$test$x, function(instance) {
  #   as.numeric(1:n %in% instance)
  # }))

  network <-
    input() +
    dense(512) +
    layer_keras("gaussian_noise", stddev = 16) +
    dense(16, activation = "sigmoid") +
    dense(512) +
    output("sigmoid")

  hash <- function(model, x, threshold = 0.5) {
    t(encode(model, x) %>% apply(1, function(r) as.integer(r > threshold)))
  }

  if (file.exists("hashing_autoencoder.tar.gz")) {
    model <- load_from("hashing_autoencoder.tar.gz")
  } else {
    model <- autoencoder(network, "binary_crossentropy") %>% train(x_train, epochs = 50)
    model %>% save_as("hashing_autoencoder.tar.gz", dir = ".")
  }

  listify <- compiler::cmpfun(function(c) {
    if (is.null(dim(c))) return(list(c))
    c %>% split(., rep(1:nrow(.), each = ncol(.)))
  })

  intercluster <- compiler::cmpfun(function(c1, c2, aggregation = mean) {
    aggregation(stringdist::seq_distmatrix(listify(c1), listify(c2), method = "cosine"))
  })

  intracluster <- compiler::cmpfun(function(cluster, aggregation = mean) {
    intercluster(cluster, cluster, aggregation)
  })

  # Measure distances among instances
  encodings <- model %>% hash(x_test)
  clusters <- list()

  for (c in 1:nrow(encodings)) {
    key <- encodings[c, ] %>% paste(collapse = "")
    clusters[[key]] <- if (is.null(clusters[[key]])) {
      x_test[c, ]
    } else {
      rbind(clusters[[key]], x_test[c, ])
    }
  }
  clusters %>% sapply(nrow) %>% summary()

  intra <- sapply(clusters, intracluster) %>% summary()

  code_n <- 16
  inter_near <- data.frame(sum = rep(0, code_n), sumsq = rep(0, code_n), count = rep(0, code_n))

  for (a in 1:(length(clusters) - 1)) {
    for (b in (a + 1):length(clusters)) {
      dist <- stringdist::stringdist(names(clusters)[a], names(clusters)[b], method = "hamming")
      dab <- intercluster(clusters[[a]], clusters[[b]])
      inter_near[dist, "sum"] <- inter_near[dist, "sum"] + dab
      inter_near[dist, "sumsq"] <- inter_near[dist, "sumsq"] + dab ** 2
      inter_near[dist, "count"] <- inter_near[dist, "count"] + 1
    }
  }

  inter_near$mean <- inter_near$sum / inter_near$count
  inter_near$sd <- inter_near$sumsq / inter_near$count - inter_near$mean ** 2
  inter_near <- rbind(c(sum = 0, count = 0, mean = intra[4], sd = 0), inter_near)

  postscript("hashing_intercluster.eps", width = 8, height = 8, paper = "special", horizontal = F)
  barplot(
    inter_near$mean,
    col = "#555555",
    names.arg = 0:code_n,
    xlab = "Hamming distance between hashes",
    ylab = "Mean intercluster cosine distance",
    xpd = FALSE
  )
  dev.off()
}
