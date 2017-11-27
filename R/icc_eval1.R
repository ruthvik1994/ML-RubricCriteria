library(jsonlite)

setwd("/home/mr/Desktop/ML-Rubric Criteria/")
csv.out <- write.csv("icc1_eval1.csv")
json.data <- fromJSON("data-json/eval1.json")
out.c <- character()
out.t <- character()
out.icc <- numeric()
ind <- 1
criteria.id <- unique(json.data[,1])
icc <- NULL
for (i in 1:length(criteria.id)){
  print (criteria.id[i])
  cur.df <- json.data[json.data$id == criteria.id[i], ]
  artifact.id <- unique(cur.df[, 3])
  list.scores <- list()
  min.num <- 100000
  max.num <- 0 
  for (j in 1:length(artifact.id)){
    list.scores[[j]] <- cur.df[cur.df$assessee_artifact_id == artifact.id[j], 4]
    min.num <- min(min.num, length(list.scores[[j]]))
    max.num <- max(max.num, length(list.scores[[j]]))
  }
  matrix.i <- data.frame()
  for (j in 1:length(list.scores)){
    l <- length(list.scores[[j]])
    if (l >= 4){
      v <- c(list.scores[[j]], rep(NA, max.num-l))
      matrix.i <- rbind(matrix.i, v)
    }
  }
  if (nrow(matrix.i) >= 3){
  icc <- ICC(matrix.i, missing = FALSE)
  out.c[ind] <- criteria.id[i]
  out.t[ind] <- cur.df[1,2]
  out.icc[ind] <- icc$results$ICC[1]
  ind <- ind+1
  }
}
output <- data.frame(out.c, out.t, out.icc)
colnames(output) <- c("Criteria ID", "Criteria", "ICC1")
write.csv(output, "icc1_eval1.csv", sep = ",")