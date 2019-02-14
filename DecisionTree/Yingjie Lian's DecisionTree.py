# Class: CS-5350
# Version: 2019/02/13
# Author: Yingjie Lian
# Hw: CS 5350 HW1 Decision Tree

import numpy as np


def checkDuplicate(L):
    arr=[]
    for i in range(len(L)):
        if L[i] not in arr:
            arr.append(L[i])

    return arr, len(arr)

def ID3(labels,size,terms,attributionSize,checkActive,type,location,depth,branch):
    
   
    if type == 'Entropy' :
        Measures = calculateEntropy(labels,size)
    elif type == 'ME' :
        Measures = calculateME(labels,size)
    else :
        Measures = calculateGI(labels,size)
    
   
    ConditionMeasures = np.zeros((np.size(attributionSize),))
    

    examples = float(np.size(labels))
    
  
    infoVariables = float(-1);
 
    attributionSplite = -1
    

    for i in checkActive:
        ConditionMeasures = 0
  
        for j in range(attributionSize[i]):
          
            locs = np.where((terms[:,i]==j))[0]
            labls = labels[locs]
            if type == 'Entropy' :
                conditionalmeasures = calculateEntropy(labls,size)
            elif type == 'ME' :
                conditionalmeasures = calculateME(labls,size)
            else :
                conditionalmeasures = calculateGI(labls,size)
            ConditionMeasures = ConditionMeasures + conditionalmeasures*float(np.size(locs))/examples
        variable_temp = Measures-ConditionMeasures
     
        if variable_temp > infoVariables :
            infoVariables = variable_temp
            attributionSplite = i
    
    branchTree = np.zeros((1,np.max(attributionSize)))
    
    attributionTree =[ [attributionSplite  ] [-1]]
    
    checkActive = np.delete(checkActive,np.where(checkActive == attributionSplite)[0][0])
    

    for j in range(attributionSize[attributionSplite]) :
   
        locs = np.where((terms[:,attributionSplite]==j))[0]
        labls = labels[locs]
  
        if np.size(labls) == 0:
            (values,counts) = checkDuplicate(labels)
            branchTree[0,j] = -1*values[np.argmax(counts)]-1
        elif np.size(np.unique(labls)) == 1 or location == depth or np.size(checkActive) == 0 :
            (values,counts) = checkDuplicate(labls)
            branchTree[0,j] = -1*values[np.argmax(counts)]-1
     
        else :
            branch = branch + 1
            branchTree[0,j] = branch
            AT, BT, branch = ID3(labls,size,terms[locs,:],attributionSize,checkActive,type,location+1,depth,branch)
            attributionTree = np.vstack((attributionTree,AT))
            branchTree = np.vstack((branchTree,BT))
            
    
    return attributionTree, branchTree, branch




def calculateGI(A,B):
    C = np.shape(A)[0]
    getCalculation = float(1)
    if C > 0 :
        for i in range(B):
            D = float(np.size(np.where((A==i))[0]))/ float(C)
            getCalculation = getCalculation - D**2
    return getCalculation

def calculateME(A,B):
    C = np.shape(A)[0]
    getME = float(0)
    maximum = float(0)
    if C > 0 :
        for i in range(B):
            D = float(np.size(np.where((A==i))[0]))/ float(C)
            if D > maximum :
                maximum = D
    getME = 1-maximum
    return getME




def calculateEntropy(A,B) :
    C = np.shape(A)[0]
    D = float(0)
    if C > 0 :
        for i in range(B):
            E = float(np.size(np.where((A==i))[0]))/ float(C)
            if E > 0 :
                D = D - E*np.log2(E)
    return D






def question2():
	TreeDepth = 6
	MeasType = 'GI'
	print('The type of this Measurement is:',MeasType)
	


	CSVfile = 'car_train.csv'

	c = 0
	with open (CSVfile , 'r') as f :
		for line in f :
        
			terms = line.strip().split(',')
			if c == 0:
          
				num_attr = np.shape(terms)[0]-1
          
				attr_vals = np.vstack((terms[0:num_attr],num_attr*['']))
       
				label_vals = terms[-1]
          
				label_size = 1
       
				labels = 0
           
				label_index = 1
          
				attr_size =  np.ones((num_attr,),dtype = int)
          
				allterms = np.zeros((num_attr,),dtype = int)
         
				terms_index = np.zeros((num_attr,),dtype = int)
			else:
           
				if terms[-1] not in label_vals:
					label_vals = np.vstack((label_vals,terms[-1]))
					label_size = label_size + 1
				label_index = np.where(terms[-1]==label_vals)[0]
				labels = np.vstack((labels,label_index))
          
				for i in range(num_attr):
               
					if terms[i] not in attr_vals[0:attr_size[i],i] :
						attr_size[i] = attr_size[i] + 1
						if attr_size[i]  > np.shape(attr_vals)[0]:
							attr_vals = np.vstack((attr_vals,num_attr*['']))
						attr_vals[attr_size[i]-1,i] = terms[i]
					terms_index[i] = np.where(terms[i]==attr_vals[0:attr_size[i]+1,i])[0][0]
				allterms = np.vstack((allterms,terms_index))
            
			c = c+1


	next_branch = 0

	attr_active = np.arange(num_attr)

	attr_tree, branches_tree, next_branch = ID3(labels,label_size,allterms,attr_size,attr_active,MeasType,1,TreeDepth,next_branch)



	num_exp = np.shape(allterms)[0]
	tree_outcome = np.zeros((num_exp,),dtype = int)
	error_count = 0;

	for i in range(num_exp) :
		tree_outcome[i] = branches_tree[0,allterms[i,attr_tree[0]]]
		while tree_outcome[i] >= 0:
			tree_outcome[i] = branches_tree[tree_outcome[i],allterms[i,attr_tree[tree_outcome[i]]]]
		tree_outcome[i] = -1*(tree_outcome[i]+1)
		if tree_outcome[i] != labels[i]:
			error_count = error_count + 1
        
	train_error = float(error_count)/float(num_exp)
	print('After calcuation, the Training Error of Car data should be:', train_error)


	CSVfile = 'car_test.csv'


	allterms_test = np.zeros((0,num_attr),dtype = int)
	labels_test = np.zeros((0,1),dtype = int)
	with open (CSVfile , 'r') as f :
		for line in f :
			terms = line.strip().split(',')
			if terms[-1] not in label_vals:
				label_vals = np.vstack((label_vals,terms[-1]))
				label_size = label_size + 1
			label_index = np.where(terms[-1]==label_vals)[0]
			labels_test = np.vstack((labels_test,label_index))
			for i in range(num_attr):
				if terms[i] not in attr_vals[0:attr_size[i],i] :
					attr_size[i] = attr_size[i] + 1
					if attr_size[i]  > np.shape(attr_vals)[0]:
						attr_vals = np.vstack((attr_vals,num_attr*['']))
					attr_vals[attr_size[i]-1,i] = terms[i]
				terms_index[i] = np.where(terms[i]==attr_vals[0:attr_size[i]+1,i])[0][0]
			allterms_test = np.vstack((allterms_test,terms_index))

   
	num_exp = np.shape(allterms_test)[0]

	tree_outcome_test = np.zeros((num_exp,),dtype = int)

	error_count = 0;


	for i in range(num_exp) :
	   tree_outcome_test[i] = branches_tree[0,allterms_test[i,attr_tree[0]]]
	   while tree_outcome_test[i] >= 0:
			tree_outcome_test[i] = branches_tree[tree_outcome_test[i],allterms_test[i,attr_tree[tree_outcome_test[i]]]]
	   tree_outcome_test[i] = -1*(tree_outcome_test[i]+1)
	   if tree_outcome_test[i] != labels_test[i]:
			error_count = error_count + 1
        
	test_error = float(error_count)/float(num_exp)
	print('After calculation, the Test Error of Car data should be:', test_error)


def question3a():
	TreeDepth = 9
	MeasType = 'Entropy'

	print('TreeDepth:', TreeDepth)
	print('Measurement Type:',MeasType)


	CSVfile = 'bank_train.csv'


	numerical_attr = np.array([0,5,9,11,12,13,14],dtype = int)

	c = 0
	with open (CSVfile , 'r') as f :
		for line in f :
        
			terms = line.strip().split(',')
			if c == 0:
		
				num_attr = np.shape(terms)[0]-1
			    
				attr_vals = np.vstack((terms[0:num_attr],num_attr*[''])).astype('U15')
			    
				label_vals = terms[-1]
				
				label_size = 1
			
				labels = 0
				
				label_index = 1
			
				attr_size =  np.ones((num_attr,),dtype = int)
			    
				allterms = np.zeros((num_attr,),dtype = int)
				
				term_arr = np.array(terms,dtype = 'U15')
			
				terms_index = np.zeros((num_attr,),dtype = int)
			else:
				
				if terms[-1] not in label_vals:
					label_vals = np.vstack((label_vals,terms[-1]))
					label_size = label_size + 1
				label_index = np.where(terms[-1]==label_vals)[0]
				labels = np.vstack((labels,label_index))
				
				for i in range(num_attr):
					
					if (terms[i] not in attr_vals[0:attr_size[i],i]) and (i not in numerical_attr) :
						attr_size[i] = attr_size[i] + 1
						if attr_size[i]  > np.shape(attr_vals)[0]:
							attr_vals = np.vstack((attr_vals,num_attr*['']))
						attr_vals[attr_size[i]-1,i] = terms[i]
					if i not in numerical_attr :
						terms_index[i] = np.where(terms[i]==attr_vals[0:attr_size[i]+1,i])[0][0]
					else :
						terms_index[i] = int(terms[i])
				allterms = np.vstack((allterms,terms_index))
            
			c = c+1
	c=0
	med = np.zeros((np.size(numerical_attr),),dtype = float)
	for i in numerical_attr :
		med[c] = np.median(allterms[:,i])
		attr_size[i]=2
		attr_vals[0:2,i] = [0,1]
		allterms[:,i] = allterms[:,i] > med[c]
		c=c+1

	
	next_branch = 0

	attr_active = np.arange(num_attr)

	attr_tree, branches_tree, next_branch = ID3(labels,label_size,allterms,attr_size,attr_active,MeasType,1,TreeDepth,next_branch)

    

	num_exp = np.shape(allterms)[0]
	tree_outcome = np.zeros((num_exp,),dtype = int)
	error_count = 0;

	for i in range(num_exp) :
		tree_outcome[i] = branches_tree[0,allterms[i,attr_tree[0]]]
		while tree_outcome[i] >= 0:
			tree_outcome[i] = branches_tree[tree_outcome[i],allterms[i,attr_tree[tree_outcome[i]]]]
		tree_outcome[i] = -1*(tree_outcome[i]+1)
		if tree_outcome[i] != labels[i]:
			error_count = error_count + 1
        
	train_error = float(error_count)/float(num_exp)
	print('Training Error:', train_error)

    
	CSVfile = 'bank_test.csv'

	
	allterms_test = np.zeros((0,num_attr),dtype = int)
	labels_test = np.zeros((0,1),dtype = int)
	with open (CSVfile , 'r') as f :
		for line in f :
			terms = line.strip().split(',')
			if terms[-1] not in label_vals:
				label_vals = np.vstack((label_vals,terms[-1]))
				label_size = label_size + 1
			label_index = np.where(terms[-1]==label_vals)[0]
			labels_test = np.vstack((labels_test,label_index))
			for i in range(num_attr):
				if (terms[i] not in attr_vals[0:attr_size[i],i]) and (i not in numerical_attr) :
					attr_size[i] = attr_size[i] + 1
					if attr_size[i]  > np.shape(attr_vals)[0]:
						attr_vals = np.vstack((attr_vals,num_attr*['']))
					attr_vals[attr_size[i]-1,i] = terms[i]
				if i not in numerical_attr :
					terms_index[i] = np.where(terms[i]==attr_vals[0:attr_size[i]+1,i])[0][0]
				else :
					terms_index[i] = int(terms[i]) > med[np.where(i == numerical_attr)[0][0]]
			allterms_test = np.vstack((allterms_test,terms_index))

    
	num_exp = np.shape(allterms_test)[0]

	tree_outcome_test = np.zeros((num_exp,),dtype = int)
	
	error_count = 0;

	for i in range(num_exp) :
	   tree_outcome_test[i] = branches_tree[0,allterms_test[i,attr_tree[0]]]
	   while tree_outcome_test[i] >= 0:
			tree_outcome_test[i] = branches_tree[tree_outcome_test[i],allterms_test[i,attr_tree[tree_outcome_test[i]]]]
	   tree_outcome_test[i] = -1*(tree_outcome_test[i]+1)
	   if tree_outcome_test[i] != labels_test[i]:
			error_count = error_count + 1
        
	test_error = float(error_count)/float(num_exp)
	print('Test Error:', test_error)




def question3b():
	TreeDepth = 1
	MeasType = 'GI'

	print('TreeDepth:', TreeDepth)
	print('Measurement Type:',MeasType)

	# Read training data
	CSVfile = 'bank_train.csv'

	#Numerical attributes
	numerical_attr = np.array([0,5,9,11,12,13,14],dtype = int)
	categ_attr = np.array([1,2,3,4,6,7,8,10,15],dtype = int)

	c = 0
	with open (CSVfile , 'r') as f :
		for line in f :
        
			terms = line.strip().split(',')
			if c == 0:
				# number of attributes
				num_attr = np.shape(terms)[0]-1
				# Each unique value is assigned an integer value
				attr_vals = np.vstack((terms[0:num_attr],num_attr*[''])).astype('U15')
				# Each unique label is assigned an integer value
				label_vals = terms[-1]
				# Number of unique labels
				label_size = 1
				# Tracks labels of the examples
				labels = 0
				# Ass0ciated index of the label value
				label_index = 1
				# Tracks number of unique value sof the atrributes
				attr_size =  np.ones((num_attr,),dtype = int)
				# Tracks atrribute value sof the examples
				allterms = np.zeros((num_attr,),dtype = int)
				#
				term_arr = np.array(terms,dtype = 'U15')
			#	allterms[numerical_attr] = term_arr[numerical_attr]
				# Associated index of the attribute value
				terms_index = np.zeros((num_attr,),dtype = int)
				for i in range(num_attr):
					if 'unknown' == terms[i]:
						attr_size[i]=0
			else:
				# Looks for new labels
				if terms[-1] not in label_vals:
					label_vals = np.vstack((label_vals,terms[-1]))
					label_size = label_size + 1
				label_index = np.where(terms[-1]==label_vals)[0]
				labels = np.vstack((labels,label_index))
				# cycles through attributes
				for i in range(num_attr):
					# Looks for new atrribute values
					if (terms[i] not in attr_vals[0:attr_size[i],i]) and (i not in numerical_attr) and ('unknown' != terms[i]):
						attr_size[i] = attr_size[i] + 1
						if attr_size[i]  > np.shape(attr_vals)[0]:
							attr_vals = np.vstack((attr_vals,num_attr*['']))
						attr_vals[attr_size[i]-1,i] = terms[i]
					if i not in numerical_attr :
						if ('unknown' != terms[i]):
							terms_index[i] = np.where(terms[i]==attr_vals[0:attr_size[i]+1,i])[0][0]
						else :
							terms_index[i] = -1
					else :
						terms_index[i] = int(terms[i])
				allterms = np.vstack((allterms,terms_index))
            
			c = c+1

	c = 0
	mode = np.zeros((np.size(categ_attr),),dtype = int)
	for i in categ_attr :
		L = allterms[:,i]
		if np.any(L== -1):
			L = np.delete(L,np.where(L == -1)[0])
		(values,counts) = checkDuplicate(L)
		mode[c] = values[np.argmax(counts)]
		allterms[np.where(-1 == allterms[:,i]),i] = mode[c]
		c = c + 1

	c=0
	med = np.zeros((np.size(numerical_attr),),dtype = float)
	for i in numerical_attr :
		med[c] = np.median(allterms[:,i])
		attr_size[i]=2
		attr_vals[0:2,i] = [0,1]
		allterms[:,i] = allterms[:,i] > med[c]
		c=c+1

	# input into ID3, defines where in the tree it is (starts at 0)
	next_branch = 0

	attr_active = np.arange(num_attr)

	attr_tree, branches_tree, next_branch = ID3(labels,label_size,allterms,attr_size,attr_active,MeasType,1,TreeDepth,next_branch)

	## Training Error

	num_exp = np.shape(allterms)[0]
	tree_outcome = np.zeros((num_exp,),dtype = int)
	error_count = 0;

	for i in range(num_exp) :
		tree_outcome[i] = branches_tree[0,allterms[i,attr_tree[0]]]
		c=0
		while tree_outcome[i] >= 0 and c <2*TreeDepth+2:
			c=c+1
			if c == TreeDepth + 2:
				print(allterms[i,:])
			tree_outcome[i] = branches_tree[tree_outcome[i],allterms[i,attr_tree[tree_outcome[i]]]]
		tree_outcome[i] = -1*(tree_outcome[i]+1)
		if tree_outcome[i] != labels[i]:
			error_count = error_count + 1
        
	train_error = float(error_count)/float(num_exp)
	print('Training Error:', train_error)

	## Test Data Error
	CSVfile = 'bank_test.csv'

	# Read test data
	allterms_test = np.zeros((0,num_attr),dtype = int)
	labels_test = np.zeros((0,1),dtype = int)
	with open (CSVfile , 'r') as f :
		for line in f :
			terms = line.strip().split(',')
			if terms[-1] not in label_vals:
				label_vals = np.vstack((label_vals,terms[-1]))
				label_size = label_size + 1
			label_index = np.where(terms[-1]==label_vals)[0]
			labels_test = np.vstack((labels_test,label_index))
			for i in range(num_attr):
				if terms[i] == 'unknown':
					terms_index[i] = mode[np.where(i== categ_attr)[0][0]]
				else:
                
					if (terms[i] not in attr_vals[0:attr_size[i],i]) and (i not in numerical_attr) :
						attr_size[i] = attr_size[i] + 1
						if attr_size[i]  > np.shape(attr_vals)[0]:
							attr_vals = np.vstack((attr_vals,num_attr*['']))
						attr_vals[attr_size[i]-1,i] = terms[i]
					if i not in numerical_attr :
						terms_index[i] = np.where(terms[i]==attr_vals[0:attr_size[i]+1,i])[0][0]
					else :
						terms_index[i] = int(terms[i]) > med[np.where(i == numerical_attr)[0][0]]
			allterms_test = np.vstack((allterms_test,terms_index))

	# Total number of test examples     
	num_exp = np.shape(allterms_test)[0]
	# Define variable ot keep track of tree outcome
	tree_outcome_test = np.zeros((num_exp,),dtype = int)
	# tracks errors
	error_count = 0;

	# Follows tree and compares example label with tree outcome
	for i in range(num_exp) :
	   tree_outcome_test[i] = branches_tree[0,allterms_test[i,attr_tree[0]]]
	   while tree_outcome_test[i] >= 0:
			tree_outcome_test[i] = branches_tree[tree_outcome_test[i],allterms_test[i,attr_tree[tree_outcome_test[i]]]]
	   tree_outcome_test[i] = -1*(tree_outcome_test[i]+1)
	   if tree_outcome_test[i] != labels_test[i]:
			error_count = error_count + 1
        
	test_error = float(error_count)/float(num_exp)
	print('Test Error:', test_error)

question2()
question3a()
question3b()
