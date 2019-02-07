# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

plot_square <- function(square, ...) {
  side <- sqrt(length(square))
  color <- FALSE

  if (side %% 1 != 0) {
    side <- sqrt(length(square) / 3)
    color <- TRUE
  }

  if (color) {
    im <- keras::array_reshape(square, c(side, side, 3), "C")
    im[im < 0] <- 0
    im[im > 1] <- 1
    colors <- rgb(im[, , 1], im[, , 2], im[, , 3])
    dim(colors) <- c(side, side)
    grid::grid.raster(colors, interpolate = FALSE)
  } else {
    im <- keras::array_reshape(square, c(side, side), "F")[, side:1]
    colors <- rgb(im, im, im)
    dim(colors) <- c(side, side)
    grid::grid.raster(colors, interpolate = FALSE)
    # image(
    #   keras::array_reshape(square, c(side, side), "F")[, side:1],
    #   xaxt = "n",
    #   yaxt = "n",
    #   col = gray((0:255) / 255),
    #   ...
    # )
  }
}

plot_sample <- function(...) {
  rows <- list(...)
  sample_size <- dim(rows[[1]])[1]
  # layout(
  #   matrix(1:(length(rows) * sample_size), byrow = F, nrow = length(rows))
  # )
  grid::grid.newpage()
  lay <- grid::grid.layout(nrow = length(rows), ncol = sample_size)
  grid::pushViewport(grid::viewport(layout=lay))

  for (i in 1:sample_size) {
    #par(mar = c(0,0,0,0) + 1)
    for (r in 1:length(rows)) {
      grid::pushViewport(grid::viewport(layout.pos.row=r, layout.pos.col=i))
      plot_square(rows[[r]][i, ])
      grid::popViewport()
    }
  }

  grid::popViewport()
}
