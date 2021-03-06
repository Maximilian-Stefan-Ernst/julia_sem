pacman::p_load(stringr)

gen_model <- function(nfact, nitem, mean_load, sd_load){
  model <- c()
  for(i in 1:nfact){
    load <- rnorm(nitem, mean_load, sd_load)
    model[i] <- 
      str_c(
        "f", 
        i, 
        "=~", 
        str_sub(
          paste(str_c(load, "*x_", i, "_", 1:nitem, " + "), collapse = ""),
          end = -3),
        "\n ")
  }
  model <- paste(model, collapse = "")
  return(model)
}

gen_model_wol <- function(nfact, nitem){
  model <- c()
  for(i in 1:nfact){
    model[i] <- 
      str_c(
        "f", 
        i, 
        "=~", 
        str_sub(
          paste(str_c("x_", i, "_", 1:nitem, " + "), collapse = ""),
          end = -3),
        "\n ")
  }
  model <- paste(model, collapse = "")
  return(model)
}

gen_model_omx <- function(nfact, nitem, data, lav_start){
  
  dataRaw <- mxData( observed=data, type="raw" )
  
  nobs = nfact*nitem
  lat_vars <- str_c("f", 1:nfact)
  observed_vars <- str_c("x_", 1:nfact, "_")
  observed_vars <- map(observed_vars, ~str_c(.x, 1:nitem))
  
  res_ind <- (nobs+1):(2*nobs)
  load_ind <- map_dbl(1:nfact, function(i){(i-1)*nitem+1})
  load_ind <- map(load_ind, ~.x:(.x+nitem-1))
  mean_ind <- (2*nobs+nfact*(nfact+1)/2+1):(2*nobs+nfact*(nfact+1)/2+nobs+nfact)
  
  # residual variances
  resVars <- mxPath( from=unlist(observed_vars), arrows=2,
                     free=TRUE,
                     values=lav_start[res_ind],
                     labels=str_c("e", 1:nobs) )
  # latent variances and covariance
  # nvoc <- nfact*(nfact+1)/2
  
  latVars <- mxPath( lat_vars, arrows=2, connect="single",
                     free=FALSE, values = rep(1, nfact))
  # latCov <- mxPath( lat_vars, arrows=2, connect="unique.bivariate",
  #                  free=TRUE, values = rep(0, nvoc-nfact) )
  
  loadings <- pmap(list(lat_vars, observed_vars, load_ind), 
                   ~mxPath(
                     from = ..1, 
                     to = ..2,
                     arrows = 1,
                     values = lav_start[..3],
                     free = rep(T, nitem),
                     labels = str_c("l_", .y)))
  
  # means
  means <- mxPath( from="one", c(unlist(observed_vars), lat_vars),
                   arrows=1,
                   free=c(rep(T, nobs), rep(F, nfact)), 
                   values=lav_start[mean_ind] )
  
  model <- 
    mxModel(
      str_c("factor_model_nfact_", nfact, "_nitem_", nitem),
      type="RAM",
      manifestVars=observed_vars,
      latentVars=lat_vars,
      dataRaw, resVars, latVars, splice(loadings), means)
  return(model)
}


induce_missing <- function(data, p){
  n_rows = nrow(data)
  n_cols = ncol(data)
  missing <- sample(c(TRUE, FALSE), 
         size = n_rows*n_cols, 
         replace = TRUE, 
         prob = c(p, 1-p))
  for (i in 1:n_rows){
    for (j in 1:n_cols){
      if (missing[(i-1)*n_cols + j]){
        data[i, j] <- NA
      }
    }
  }
  return(data)
}
