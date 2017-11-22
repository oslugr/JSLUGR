
#paquete TSclust contiene distancias entre series, métodos de cluster y validación
#datasets de ejemplo
library(TSclust)

#cargamos un dataset
data(synthetic.tseries)

#pintamos unas series
plot(synthetic.tseries[,c(1,4,7)])

pdf("syntheticplot.pdf", 16*0.5, 9*0.5)
plot(synthetic.tseries[,c(1,4,7)], main="")
dev.off()



#aplicamos una distancia usando diss():
#la funcion diss calcula una distancia de TSclust
#a todos los pares posibles de series en nuestros datos
#y devuelve un objeto diss
#que suelen usar los métodos de cluster
dCEPS <- diss(synthetic.tseries, "AR.LPC.CEPS")
#tomamos una distancia basada en modelos autoregresivos

#con el objeto diss dCEPS podemos hacer cluster
#por ejemplo cluster jerárquico
#se presta a la comparación visual

hcCEPS <- hclust(dCEPS)


#apliquemos una distancia de datos en bruto como la Euclídea
dEUCL <- diss(synthetic.tseries, "EUCL")

#aplicamos el mismo método de cluster que con la distancia basada en modelos
hcEUCL <- hclust(dEUCL)

#comparemos visualmente las dos
plot(hcEUCL)
plot(hcCEPS)



#las distancias en TSclust tienen su funcion individual
#documentada y con referencia al artículo en el que fueron publicadas
?diss.AR.MAH

#esta distancia basada en modelos produce un p-valor
#del contraste que compara si dos series
#tienen el mismo modelo
dMAH <- diss(synthetic.tseries, "AR.MAH")

#TSclust tiene un método de cluster basado en p-valores
#que agrupa fijando un nivel de significación

#interés por interpretabilidad
#en función de la significación crea automáticamente
#los clusters sin evidencias suficientes para separar

pvalues.clust(dMAH$p_value, 0.01)

pvalues.clust(dMAH$p_value, 0.05)


#cargar un dataset con series
#de diferentes dominios
#ECG, Sensores de movimiento,
#Demanda eléctrica...
data("paired.tseries")

#usamos la distancia PDC
#del paquete pdc
#incluida en TSclust por conveniencia
dPDC = diss(paired.tseries, "PDC", m=5,t=8)

plot(hclust(dPDC))


data("interest.rates")

#aplicamos cluster basado en predicción
#el método de predicción es un
#autoregresivo no-paramétrico
#que acepta transformaciones clásicas
#diferenciación y logaritmos

#en este caso aplicamos las dos transformaciones
diffs <- rep(1, ncol(interest.rates))
logs <- rep(TRUE, ncol(interest.rates))

?diss.PRED

#comparamos las predicciones a un horizonte de 6 años
dPRED <- diss(interest.rates, "PRED", h=6, B=1200, logarithms=logs,
              differences=diffs, plot=TRUE)

plot(hclust(dPRED$dist ))




#probamos con el datasaet inicial
#de series temporales simuladas

#creamos el valor de verdad "real"
tssynthetic_truth <- rep(1:6, each=3)

#aplicamos cluster usando PAM
pamCEPS <- pam(dCEPS, k=6)$clustering
pamEUCL <- pam(dEUCL, k=6)$clustering

#medida de evaluación implementada en TSclust
cluster.evaluation(tssynthetic_truth, pamCEPS)
[1] 0.875
cluster.evaluation(tssynthetic_truth, pamEUCL)
[1] 0.6944444

#el paquete fpc contiene medidas más populares
#de evaluación
#con valor de verdad como sin el
library(fpc)

cluster.stats(dCEPS, clustering=pamCEPS,
               alt.clustering=tssynthetic_truth )
$corrected.rand
[1] 0.7702703
$avg.silwidth
[1] 0.6442922
cluster.stats(dEUCL, clustering=pamEUCL,
              alt.clustering=tssynthetic_truth )
$corrected.rand
[1] 0.5142857
$avg.silwidth
[1] 0.1225102

####################
### CLASIFICACTION #
#####################

#obtenemos un dataset de ejemplo
download.file("http://timeseriesclassification.com/Downloads/ECG200.zip","ECG200.zip")
unzip("ECG200.zip")

library(foreign)

trainset <- as.matrix(read.arff("ECG200/ECG200_TRAIN.arff"))
class(trainset) <- "numeric"
trainclasses <- trainset[, ncol(trainset)]
trainset <- trainset[,-ncol(trainset)]

#Se puede lograr buenos resultados con ensembles de distancias
#realmente no solemos necesitar un método que funcione bien en todos los
#tipos de series temporales, por lo que quizas con una distancia apropiada llegue

dDTW <- diss(trainset, "DTWARP")
#calculamos crossvalidation accuracy con TSclust
loo1nn.cv(dDTW, trainclasses)

dCORT <- diss(trainset, "CORT")
loo1nn.cv(dCORT, trainclasses)

#con TSclust y TSdist tenemos una gama de distancias que permite reproducir
#métodos publicados
dEDR <- proxy::dist(trainset, EDRDistance, epsilon=0.1)
loo1nn.cv(dEDR, trainclasses)


#cargamos el conjunto de test
testset <- as.matrix(read.arff("ECG200/ECG200_TEST.arff"))
class(testset) <- "numeric"
testclasses <- testset[, ncol(testset)]
testset <- testset[,-ncol(testset)]

#1-NN para las 3 distancias de ejemplo
predclassesDTW <- apply(testset, 1, function (xts) {
  distances <- apply(trainset, 1, function(yts) diss.DTWARP(xts, yts))
  trainclasses[which.min(distances)]
})


predclassesCORT <- apply(testset, 1, function (xts) {
  distances <- apply(trainset, 1, function(yts) {diss.CORT(xts, yts)})
  trainclasses[which.min(distances)]
})


predclassesEDR <- apply(testset, 1, function (xts) {
  distances <- apply(trainset, 1, function(yts) {EDRDistance(xts, yts, 0.1)})
  trainclasses[which.min(distances)]
})

#ensemble, majority voting ponderado por loo1nn crossvalidation
#código es mejorable :)
C <- unique(trainclasses)

C1 <- (predclassesEDR==C[1]) * loo1nn.cv(dEDR, trainclasses) +
  (predclassesCORT==C[1]) * loo1nn.cv(dCORT, trainclasses) +
  (predclassesDTW==C[1]) * loo1nn.cv(dDTW, trainclasses)

C2 <- (predclassesEDR==C[2]) * loo1nn.cv(dEDR, trainclasses) +
  (predclassesCORT==C[2]) * loo1nn.cv(dCORT, trainclasses) +
  (predclassesDTW==C[2]) * loo1nn.cv(dDTW, trainclasses)

#resultados clasificacion, se puede ver que CORT produce mejores resultados
#que el ensemble y próximos al máximo en la literatura
mean(C[((C2 - C1) > 0)  + 1] == testclasses)

######################################################
#contraste de hipotesis aplicado a series temporales #
######################################################

#en muchos casos, los test estandar consiguen poca potencia en series temporales
#hay tests que aceptan una familia de distancias
#por otro lado, puede haber diferencias entre poblaciones que no consideramos de interes
#por ejemplo escala, o cambio de fase,
#por lo que hay distancias que son invariantes a estos cambios
#un test combinado con esta distancia permitiria capturar todos sus beneficios

#usamos el energy statistic, que acepta semimetricas de tipo negativo
library(energy)

#compararemos el dataset de series sinteticas autoregresivas
#la primera mitad contra la segunda mitad
dSPEC = diss(synthetic.tseries, "PER")
energy::eqdist.etest(dSPEC, c(9,9), R=10000)


dEUCL = diss(synthetic.tseries, "EUCL")
energy::eqdist.etest(dEUCL, c(9,9), R=10000)

#vemos que la potencia aplicando una distancia de series temporales es muy superior