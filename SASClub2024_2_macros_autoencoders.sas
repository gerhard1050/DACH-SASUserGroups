/*-----------------------
  MACROS FOR AUTOENCODERS
  -----------------------*/

/*
List of contained macros:

  ann_input_output		Identification of column names storing-input and output-layer nodes
  ae_rerr				Reconstruction error of an Autoencoder
*/




/*--------------------------------------------------------------------------*/

/*----------------
  ANN_INPUT_OUTPUT
  ----------------*/

/*~

# Summary
This macro allows for identification of the names of the nodes in the input respectively the output layer
of a trained Autoencoder (form neuralNet.annTrain). The names are stored in two macro variables (space-separated).
In addition, also the original input feature names are written to a macro variable.


# Usage
In order to compute the reconstruction error of an Autoencoder, we need to know which nodes in the table of
predictions belong to the input layer and which are the corresponding nodes in the output layer. This macro
takes as an input a CAS table storing a trained autoencoder and writes the names of all input- respectively
output-nodes to two user-specified macro variables.



# Parameters
* modelTable		= A SAS table storing a trained Autoencoder (from neuralNet.annTrain)
* mvar_nodes_in		= Macro variable storing the names of nodes in the input layer
* mvar_nodes_out	= Macro variable storing the names of nodes in the output layer
* mvar_feat_names	= Macro variable storing the original names of the input features



# Result
Three macro variables named according to mvar_nodes_in, mvar_nodes_out and mvar_feat_names storing the names of nodes in the
input and output layers respectively the original names of the input features.



# Example
The following code requires a CAS table storing a trained Autoencoder (from neuralNet.annTrain) in the caslib casuser.
%ann_input_output(casuser.autoencoder_trained,
				  mvar_nodes_in=_input_nodes,
				  mvar_nodes_out=_output_nodes,
				  mvar_feat_names=_feat_names);



Authors: Phillipp Gnan

~*/

%macro ann_input_output(modelTable, mvar_nodes_in=_input_nodes, mvar_nodes_out=_output_nodes, mvar_feat_names=_feat_names);
	%local helper_mvar_tmp;	/*only defined to avoid proc sql warnings (would warn that we define fewer mvars than selected columns), and because we need to ensure ordering by _NodeID_*/
	%global &mvar_nodes_in. &mvar_nodes_out. &mvar_feat_names.;
	
	proc sql noprint;
		/*input layer nodes*/
		select distinct catx("", "_Node_", _NodeID_), _VarName_, _NodeID_
			   into :&mvar_nodes_in. separated by " ", :&mvar_feat_names. separated by " ", :helper_mvar_tmp separated by ""
			from &modelTable.
			where _LayerID_ = 0
			order by _NodeID_;
		/*output layer nodes*/
		select distinct catx("", "_Node_", _NodeID_), _NodeID_
			   into :&mvar_nodes_out. separated by " ", :helper_mvar_tmp separated by ""
			from &modelTable.
			having _LayerID_ = max(_LayerID_)
			order by _NodeID_;
	quit;
%mend ann_input_output;





/*--------------------------------------------------------------------------*/

/*----------------------------
  AUTOENC_RECONSTRUCTION_ERROR
  ----------------------------*/

/*~

# Summary
Macro to compute the squared difference between every input node and the corresponding output node
and take the mean across nodes within each observation.

# Usage
We compute the reconstruction error by comparing nodes in the input layer to the respective nodes in the output layer.
When training the autoencoder, we need to generate two mvars _input_nodes and _output_nodes containing the colnames of nodes in the input respectively output layers.

	- The first m nodes (_Node_0 to _Node_[m-1]) correspond to the nodes of the input layer, i.e. transformed (=standardized) input features.
	- The last m nodes (_Node_[N-m] to _Node_[N-1]) correspond to the nodes of the output layer (N is the overall number of nodes, including input, hidden and output layers)
	- All nodes in between correspond to the hidden layers, from the first to the last hidden layer



# Parameters
* pred_table		output table from neuralNet.annScore action, with listnode argument set to "ALL" (i.e. all nodes from input to output layer are included)
* input_nodes		space-separated list of colnames in pred_table corresponding to input-layer nodes
* output_nodes		space-separated list of colnames in pred_table corresponding to output-layer nodes
* feat_names		space-separated list of input feature names (or new desired names): if provided, the feature-level reconstruction error is returned as well
* table_out			name of output table (provided in format libname.tablename)
* keep				space-separated list of columns to keep from from pred_table (in addition to newly computed columns for the reconstruction error)



# Result
A CAS table named according to the table_out parameter. Containing the aggregate reconstruction error, the feature-level reconstruction errors (if feat_names are provided)
and any columns from the input table (parameter pred_table) specified with the keep parameter.



Authors: Phillipp Gnan

~*/




%macro ae_rerr(pred_table, input_nodes, output_nodes, feat_names, table_out, keep);
	%local _i input_i output_i name_i;

	/*Feature-level reconstruction error plus aggregate error*/
	%if &feat_names. ^= %then %do;
		%put NOTE: Writing feature-level reconstruction error to table. To disable, provide no argument for feat_names.;

		data &table_out.;
			set &pred_table.(keep=&input_nodes. &output_nodes. &keep.);

			/*Feature-level reconstruction error*/
			%do _i=1 %to %sysfunc(countw(&input_nodes., %str( ))); 
					  	%let input_i = %scan(&input_nodes., &_i., %str( ));		/*get ith input-layer node*/
						%let output_i = %scan(&output_nodes., &_i., %str( ));	/*get ith output-layer node*/
						%let name_i = %scan(&feat_names., &_i., %str( ));		/*get ith input feature name*/

						rerr_&name_i. = &input_i. - &output_i.;				/*reconstruction error for feature i*/
						label rerr_&name_i. = "Feature-level Autoencoder reconstruction error (Input - Output)";
			%end;

			/*Aggregate reconstruction error*/
			ae_rerr=mean(
						 %do _i=1 %to %sysfunc(countw(&input_nodes., %str( ))); 
							%let name_i = %scan(&feat_names., &_i., %str( ));		/*get ith input feature name*/
							rerr_&name_i.**2,
						 %end;
						.);															/*add missing, just to catch the last ","*/
			label ae_rerr = "Autoencoder reconstruction error (average squared error)";

			drop &input_nodes. &output_nodes.;
		run;	
	%end;

	%else %do;
		data &table_out.(keep=ae_rerr &keep.);
			set &pred_table.;
			ae_rerr=mean(
						 %do _i=1 %to %sysfunc(countw(&input_nodes., %str( ))); 
						  	%let input_i=%scan(&input_nodes., &_i., %str( ));		/*get ith input-layer node*/
							%let output_i=%scan(&output_nodes., &_i., %str( ));		/*get ith output-layer node*/
							(&input_i. - &output_i.)**2,
						 %end;
						.);
			label ae_rerr = "Autoencoder reconstruction error (average squared error)";															/*add missing, just to catch the last ","*/
		run;
	%end;																		
%mend ae_rerr;

