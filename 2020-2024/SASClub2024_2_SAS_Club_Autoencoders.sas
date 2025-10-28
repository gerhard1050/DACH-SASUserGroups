/*--------------------------
  AUTOENCODERS - TOY EXAMPLE
  --------------------------

  Author: Phillipp Gnan - phillipp.gnan@bmf.gv.at
*/



/*--------------------------------------------------------------------------*/

/*-------------
  SESSION SETUP
  -------------*/

cas my_session;		/*start session*/

caslib _all_ assign; /*assign librefs to all existing caslibs*/



/*Python session is optional - only necessary for visualizations*/
proc python restart;
	submit;
import pandas as pd
import matplotlib.pyplot as plt
	endsubmit;
run;



/*--------------------------------------------------------------------------*/

/*-----------------
  GENERATE TOY DATA
  -----------------

  In this example, we are using the well-known iris flower data set available through the SASHELP library.
  The dataset consists of 50 samples from each of three species of Iris flowers: Iris Setosa, Iris Versicolor, and Iris Virginica.
  Four features were measured for each sample:

	- Sepal length (cm)
	- Sepal width (cm)
	- Petal length (cm)
	- Petal width (cm)
*/

data casuser.iris;
	set sashelp.iris;
	ID=_n_;							/*generate ID*/
	tg=0;
	if mod(_n_, 10) = 0 then do;	/*generate some outliers*/
		sepallength = 0;
		tg=1;						/*target: marker for outliers*/
	end;
run;







/*--------------------------------------------------------------------------*/

/*---------------------------
  TRAIN-VALIDATION-TEST SPLIT
  ---------------------------*/


/*Generate group assignment*/
proc cas;

	sampling.stratified result=r/
		table={name="iris",
			   groupby={"Species"}}					/*groupby lists all variables used for stratification, here our target variable*/
		samppct=60									/*60% train (group 1)*/
		samppct2=20									/*20% validation (group 2)*/
		partInd=TRUE								/*add a partition indicator column (_PartInd_)*/
		seed=123
		output={casOut={name="train_valid_test",	/*store group assignment in casuser.train_valid_test*/
				replace=TRUE},		
				copyVars="ALL"};
	run;

	table.fetch /                                   /*fetch train_valid_test*/
		table="train_valid_test";
	run;

	print r.STRAFreq; run;							/*Print result summary*/

quit;


/*Store each sample in a separate table*/
data casuser.train casuser.valid casuser.test;
	set	casuser.train_valid_test;
	if _partInd_=1 then output casuser.train;
	else if _partInd_=2 then output casuser.valid;
	else output casuser.test;
run;


/*delete tables no longer needed*/
proc delete data=casuser.iris casuser.train_valid_test; run;




/*--------------------------------------------------------------------------*/

/*--------------------
  AUTOENCODER TRAINING
  --------------------

  If no target is specified, the annTrain action automatically trains an Autoencoder.
  - To include nominal inputs, include them in inputs={} and specify the additional argument nominals={} to declare them as nominals (e.g. nominals={{name="mynominal1"}, {name="mynominal2"}})*/
proc cas;
	neuralNet.annTrain result=r /
	/*INPUTS*/
		table={name="train"}											/*input table*/
		inputs={"sepallength","sepalwidth","petallength","petalwidth"}	/*features*/
		std="MIDRANGE"													/*standardization of scalar features: (x-midrange)/halfrange*/
	/*ARCHITECTURE*/
		hiddens={2, 1}													/*number of neurons in each hidden layer (this implicitly also defines the number of layers)*/
		combs={"LINEAR"}												/*combination functions used in each hidden layer*/
		acts={"RECTIFIER"}												/*activation function in each hidden layer*/
		targetAct="IDENTITY"											/*activation function in output layer*/
	/*OPTIMIZER*/
		seed=123
		errorFunc="NORMAL"												/*error function, i.e. objective function (excluding potential L1/L2 terms) to minimize with SGD*/
		randDist="UNIFORM"												/*distribution for sampling initial NN connection weights*/
		scaleInit=1														/*scaling factor for *initial* NN connection weights relative to the number of nodes in previous layer; initially drawn weights are scaled to range [-scaleInit/sqrt(n_nodes), scaleInit/sqrt(n_nodes)]*/
		nloOpts={														/*options for the solver used for optimization*/
			algorithm="SGD",											/*SGD also takes ADAM arguments in sgdOpt (e.g. momentum); trying to directly specify the algorithm as "ADAM" currently does not work for me (Viya version 2024/07)*/
			optmlOpt={maxIters=200, fConv=1e-10},						/*maximum number of iterations, stopping criterion: stop if objective function changes by less than fConv*/
/* 			lbfgsOpt={numCorrections=6}, */
			sgdOpt={adaptiveDecay=0.99,
					adaptiveRate=True,
					learningRate=0.1,
					miniBatchSize=5,
					momentum=0.95},
			validate={frequency=1,										/*frequency=1: validation occurs at every epoche*/
					  stagnation=3}}									/*terminate early if validation error increases for 4 consecutive epochs*/
		validTable={name="valid"}										/*validation set to assess model during training for early stopping*/
	/*OUTPUT*/
		casout={name="autoencoder_trained", replace=True}
		encodename=True													/*whether to encode the variable names in the output cas table*/
		;


	describe r;

	print r.ConvergenceStatus.compute("Epochs", r.OptIterHistory.nrows);
	if r.OptIterHistory.nrows <= 20 then print r.OptIterHistory[1:r.OptIterHistory.nrows];
		else print r.OptIterHistory[(1:10) + ((r.OptIterHistory.nrows-9):r.OptIterHistory.nrows)];
quit;




/* Get the names of input nodes and output nodes from the table storing the
   trained Autoencoder using the custom %ann_input_output macro.*/


%ann_input_output(casuser.autoencoder_trained,		/*trained Autoencoder*/
				  mvar_nodes_in=_input_nodes,		/*name of mvar storing input layer nodes*/
				  mvar_nodes_out=_output_nodes,		/*name of mvar storing output layer nodes*/
				  mvar_feat_names=_feat_names);		/*name of mvar storing the original names of the input features*/

%put &=_input_nodes;
%put &=_output_nodes;
%put &=_feat_names;



/*--------------------------------------------------------------------------*/

/*-------------------
  AUTOENCODER SCORING
  -------------------*/

proc cas;
	neuralNet.annScore /
		table={name="test"}
		modelTable={name="autoencoder_trained"}
		copyvars={"id"}								/*keep ID variable*/
		listnode="ALL"								/*includes nodes for the (standardized) input, hidden and output layers in the output table*/

		casOut={name="pred_test", replace=TRUE}
		;	
quit;

/* title "casuser.pred_test"; */
/* proc print data=casuser.pred_test; run; */






/*--------------------------------------------------------------------------*/

/*--------------------
  RECONSTRUCTION ERROR
  --------------------

  We compute the reconstruction error by comparing nodes in the input layer to the respective nodes in the output layer.
  We previously generated two mvars _input_nodes and _output_nodes containing the colnames of nodes in the input respectively output layers.

	- The first m nodes (_Node_0 to _Node_[m-1]) correspond to the nodes of the input layer, i.e. transformed (=standardized) input features.
	- The last m nodes (_Node_[N-m] to _Node_[N-1]) correspond to the nodes of the output layer (N is the overall number of nodes)
	- All nodes in between correspond to the hidden layers, from the first to the last hidden layer

*/


%ae_rerr(pred_table=casuser.pred_test,															/*Autoencoder predictions*/
		 input_nodes=&_input_nodes., output_nodes=&_output_nodes., feat_names=&_feat_names.,
		 table_out=casuser.rerr,																/*output table*/
		 keep=id);																				/*additional columns to keep*/


/*Test data average reconstruction error*/
proc sql;
	select mean(ae_rerr) as ae_rerr_avg label="Test data: Average reconstruction error (i.e. average MSE)"
	from casuser.rerr;
quit;


/*Merge reconstruction error to original table*/
data casuser.test_with_rerr;
	merge casuser.test casuser.rerr;
		by ID;
run;

/*Compare reconstruction error for tg=0 vs tg=1*/
proc sql;
	select tg, mean(ae_rerr) as average_rerr, count(*) as N
	from casuser.test_with_rerr
	group by tg;
quit;



/*Visualizations of reconstruction error*/
proc python; submit;
d = SAS.sd2df("casuser.test_with_rerr") # LOAD DATA


## NORMAL VS OUTLIER AGGREGATE RERR
fig, ax = plt.subplots()
ax.hist(d.loc[d.tg==0, "ae_rerr"], label="no", bins=10)
ax.hist(d.loc[d.tg==1, "ae_rerr"], label="yes", bins=50)
ax.legend(title="Outlier")
ax.set_title("Normal vs outlier: reconstruction error")
ax.set_xlabel("Reconstruction error (MSE)")
ax.set_ylabel("Count")
SAS.pyplot(fig)


## NORMAL VS OUTLIER RERR BY FEATURE
rerr_by_feat = (d
	.filter(regex=r"^rerr|^tg$")
	.groupby("tg")
	.apply(lambda g: (g**2).mean(axis=0) )
	.drop("tg", axis=1)
	)

fig, ax = plt.subplots()
rerr_by_feat.plot.bar(ax=ax)
ax.set_title("Average reconstruction error by feature")
ax.set_xlabel("Outlier")
ax.set_ylabel("Average reconstruction error (MSE)")
ax.set_xticklabels(["no", "yes"], rotation=0)
SAS.pyplot(fig)
	endsubmit;
run;


/*Visualizations of outliers in train data*/
proc python; submit;
d = SAS.sd2df("casuser.train") #LOAD DATA

## GET FEATURES
feats = d.columns.str.extract(r"(Sepal.*|Petal.*)", expand=False).dropna()

## PAIRWISE SCATTER PLOTS
fig, ax = plt.subplots()
ax = pd.plotting.scatter_matrix(d[feats], ax=ax)

for i in range(len(feats)):
	for j in range(len(feats)):
		if i==j:
			continue
		y = feats[i]
		x = feats[j]
		if "SepalLength" in {x, y}:
			alpha=1
		else:
			alpha=0.3
		ax[i,j].scatter(d.loc[d.tg==1,x], d.loc[d.tg==1,y],
						s=6, color="orange", alpha=alpha)

fig.suptitle("Training Data: Pairwise Scatter Plots")

SAS.pyplot(fig)
	endsubmit;
run;









/*--------------------------------------------------------------------------*/

/* TERMINATE CAS SESSION */
/* cas my_session terminate; */
