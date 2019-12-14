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
  library(mldr.datasets)
  dataset <- bibtex()
  y <- dataset$dataset[, dataset$labels$index]
  dataset <- dataset$dataset[, dataset$attributesIndexes]
  dataset <- apply(dataset, 2, as.numeric)

  in_test <- sample(1:nrow(dataset), floor(nrow(dataset)/5)) # 80-20 split

  x_train <- as.matrix(dataset[-in_test, ])
  x_test <- as.matrix(dataset[in_test, ])
  y_test <- as.matrix(y[in_test, ])
  rownames(x_test) <- paste0("T", 1:nrow(x_test))
  rownames(y_test) <- rownames(x_test)

  # x_train <- matrix(0, nrow = length(dataset$train$x), ncol = n)
  # x_train <- t(sapply(dataset$train$x, function(instance) {
  #   as.numeric(1:n %in% instance)
  # }))
  # x_test <- t(sapply(dataset$test$x, function(instance) {
  #   as.numeric(1:n %in% instance)
  # }))
  code_n <- 7

  network <-
    input() +
    dense(512) +
    dense(code_n, activation = "linear") +
    layer_keras("gaussian_noise", stddev = 16) +
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
  x_test_hash <- x_test
  x_test_hash$hash <- apply(encodings, 1, function(r) paste0(r, collapse = ""))
  rownames(encodings) <- rownames(x_test)
  clusters <- list()

  for (c in 1:nrow(encodings)) {
    key <- encodings[c, ] %>% paste(collapse = "")
    clusters[[key]] <- c(clusters[[key]], list(x_test[c, ]))
  }
  clusters %>% sapply(length) %>% summary()
  clusters <- clusters %>% map(compose(t, as.data.frame))

  intra <- sapply(clusters, intracluster) %>% summary()

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
  inter_near[1, "mean"] <- mean(intra)

  distances <- expand.grid(h1 = names(clusters), h2 = names(clusters))
  distances <- cbind(
    distances,
    hamming = stringdist::stringdist(distances$h1, distances$h2, method = "hamming"),
    mean_cosine = sapply(1:nrow(distances), function(i) {
      intercluster(clusters[[distances$h1[i]]], clusters[[distances$h2[i]]])
    })
  )

  library(ggplot2)
  postscript("hashing_intercluster.eps", width = 6, height = 6, paper = "special", horizontal = F)
  ggplot(distances, aes(x=as.character(hamming), y=mean_cosine)) +
    geom_boxplot(outlier.shape = 20, outlier.colour = "darkgray") +
    stat_summary(fun.y=mean, geom="point", shape=18, size=4, color="blue") +
    xlab("Hamming distance among hashes") +
    ylab("Cosine distance among instances")
  dev.off()

  idf <- function() {
    numtest <- apply(x_test, 2, as.numeric)
    log(length(clusters) / colSums(numtest))
  }
  IDF <- idf()

  tf <- function(n) {
    count <- if (is.null(dim(clusters[[n]])) || dim(clusters[[n]])[1] == 1) {
      as.numeric(clusters[[n]])
    } else {
      numcl <- apply(clusters[[n]], 2, as.numeric)
      colSums(numcl)
    }
    # log-scaled frequency:
    log(1+count)
  }

  # Print some instances from the same cluster
  tf_idf <- function(n) {
    tfidf <- tf(n)*IDF
    # print(head(tfidf[order(tfidf, decreasing = T)]))
    invisible(tfidf)
  }
  # calcular tf-idf?

  # Print hashes and relevant words in gray code order
  for (n in 1:(2**7-1)) {
    gray <- bitwXor(n, bitwShiftR(n, 1))
    graybin <- R.utils::intToBin(gray)
    name <- paste0(paste0(rep("0", 7-nchar(graybin)), collapse=""), graybin)
    # print(name)
    if (name %in% names(clusters)) {
      ti <- tf_idf(which(names(clusters) == name))
      relevant <- head(names(ti)[order(ti, decreasing = T)], n=6)
      cat(name, " & ", paste0(relevant, collapse = ", "), "\\\\\n")
    }
  }

  #------------Evaluation------------------
#
#   precision <- function(cl) {
#     rownames(cl)
#   }
}
