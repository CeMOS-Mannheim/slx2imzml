checkReady <- function(app) {
  if (isTruthy(app$slxPath) &
    isTruthy(app$selRegions))
  #  & isTruthy(app$selFeatureLists))
  {
    app$ready <- TRUE
  } else {
    app$ready <- FALSE
  }
}

checkReady <- function(app) {
  if (isTruthy(app$slxPath) &
    isTruthy(app$selRegions) &
    isTruthy(app$selFeatureLists)) {
    app$ready <- TRUE
  } else {
    app$ready <- FALSE
  }
}
