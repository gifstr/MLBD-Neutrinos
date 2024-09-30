import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import time

class DataPrep():
    '''
    This class prepares the data for the regression model.

    Attributes:
        table: A pandas dataframe containing the table of values corresponding to the energy level.
    '''
    def __init__(self, table):
        self.table = table


    def sets(self):
        '''
        This function splits the table into more tables for each component of the tensor and for training and testing.
        It rearranges the tables structure into one that is suitable for the train/test split

        The tables are then stored as attributes of the class and they are:
        - targets: A dictionary containing the real and imaginary parts of each component of the tensor.
        - re: A dictionary containing the real part of each component of the tensor.
        - im: A dictionary containing the imaginary part of each component of the tensor.
        - kinematics: A dataframe containing the kinematic variables (common for all components).
        - tensor: A dataframe containing the real and imaginary parts of each component of the tensor.
        - Real: A dataframe containing the real parts of each component of the tensor.
        - Imaginary: A dataframe containing the imaginary parts of each component of the tensor.
        - all_Re: A dataframe containing the kinematic variables and the real parts of each component of the tensor.
        - all_Im: A dataframe containing the kinematic variables and the imaginary parts of each component of the tensor.
        - all: a dataframe containing the entire rearranged table (kinematics + tensor components)

        Returns:
            self: The class object with the new tables as attributes.    
        '''
        self.targets = {}
        self.re = {}
        self.im = {}
        
        #iterates over all tensor components
        for i in range(10):
            self.targets[i] = self.table[self.table['ij'] == i][['Re','Im']].reset_index(drop=True)
            self.targets[i].rename(columns = {'Re': 'Re%g'%i,'Im': 'Im%g'%i}, inplace=True)
            self.re[i] = self.targets[i]['Re%g'%i].reset_index(drop=True)
            self.im[i] = self.targets[i]['Im%g'%i].reset_index(drop=True)

        self.kinematics = self.table[self.table['ij'] == 0].drop(['ij','Re','Im'], axis=1).reset_index(drop=True)
        self.tensor = pd.concat(self.targets.values(), axis=1)
        self.tensor = self.tensor.dropna().reset_index(drop=True)
        self.Real = pd.concat(self.re.values(), axis=1)
        self.Imaginary = pd.concat(self.im.values(), axis=1)
        self.all_Re = pd.concat([self.kinematics, self.Real], axis=1)
        self.all_Im = pd.concat([self.kinematics, self.Imaginary], axis=1)
        self.all = pd.concat([self.kinematics, self.tensor], axis=1)
        return self
    

    def contours(self,x,y,z):
        '''
        This function plots the tensor component value for each component on a 2D plot of the phase space.
        The color of the points represents the component value.

        Args:
            x: A string that specifies the kinematic variable that is on the x-axis.
            y: A string that specifies the kinematic variable that is on the y-axis.
            z: A string that specifies the part of the tensor (Re or Im) to be plotted.
        '''
        kin = {'T', 'θ', 'p3', 'E'}
        axes = set([x,y]) #the kinematics of the x and y axes
        const = list(kin-axes) #the kinematics that are kept constant
        const1 = self.all[const[0]].value_counts().idxmax() #finds the most common value of the first constant kinematic (can be changed here)
        const2 = self.all[const[1]].value_counts().idxmax() #finds the most common value of the second constant kinematic (can be changed here)
        #a dataframe containing the values that will be plotted
        all_plot = self.all.loc[(self.all[const[0]] == const1) & (self.all[const[1]] == const2)] 
        #setting up common colorbar across all subplots
        cm = 'plasma'
        cbar = plt.cm.ScalarMappable(cmap=cm)
        cbar.set_array([])
        z_min, z_max = float('inf'), float('-inf')
        plt.figure(figsize=(15,6))
        #iterating over all tensor components
        for i in range(10):
            plt.subplot(2,5,i+1)
            X,Y,Z = all_plot[x], all_plot[y], all_plot['%s%i'%(z,i)]
            z_min = min(z_min, Z.min()) #for the common colorbar
            z_max = max(z_max, Z.max()) #for the common colorbar
            plt.suptitle('Constants: %s=%g and %s=%g'%(const[0],const1,const[1],const2))
            plt.scatter(X,Y, c=Z, cmap=cm, marker='s', s = 5)
            plt.title('Component %i'%i)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.tight_layout()
        
        #adjusting colorbar
        cbar.set_clim(z_min, z_max)
        plt.subplots_adjust(right=0.9)
        cbar_ax = plt.gcf().add_axes([0.92, 0.09, 0.02, 0.8])
        plt.colorbar(cbar, cax=cbar_ax, label = 'Tensor Value')
        plt.show()


    def split(self, scaling):
        '''
        This function splits the tables into training, validation and testing sets and scales/transforms them if needed (Yeo-Johnson power transform).

        Args:
            scaling: A boolean that determines if the data should be scaled/transformed or not.

        Returns:
            X_train: A dataframe containing the kinematic variables for the training set.
            y_train: A dataframe containing the real and imaginary parts of each component of the tensor for the training set.
            X_val: A dataframe containing the kinematic variables for the validation set.
            y_val: A dataframe containing the real and imaginary parts of each component of the tensor for the validation set.
            X_test: A dataframe containing the kinematic variables for the testing set.
            y_test: A dataframe containing the real and imaginary parts of each component of the tensor for the testing set.
            X_temp: A dataframe containing the kinematic variables for the temporary testing set.
            y_temp: A dataframe containing the real and imaginary parts of each component of the tensor for the temporary testing set.
        '''
        #splits the data randomly into training (70%) and the temporary test (30%) subsets
        X_train, X_temp, y_train, y_temp = train_test_split(self.kinematics, self.tensor, test_size=0.3, random_state=42, shuffle=True)
        #splits the temporary test set randomly into testing (15%) and validation (15%) subsets
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)

        #no scaling/transform is applied
        if scaling == False:
            return X_train.reset_index(drop=True), y_train.reset_index(drop=True), X_val.reset_index(drop=True), y_val.reset_index(drop=True), X_test.reset_index(drop=True), y_test.reset_index(drop=True), X_temp.reset_index(drop=True), y_temp.reset_index(drop=True)

        #apply yeo-johnson power transformation
        #scaling/trasnformation method can be changed here
        else:
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            
            y_train = pd.DataFrame(data = scaler.fit_transform(y_train), columns = y_train.columns)
            y_test = pd.DataFrame(data = scaler.transform(y_test), columns = y_test.columns)
            y_val = pd.DataFrame(data = scaler.transform(y_val), columns = y_val.columns)
            y_temp = pd.DataFrame(data = scaler.transform(y_temp), columns = y_temp.columns)
    
            return X_train, y_train, X_val, y_val, X_test, y_test, X_temp, y_temp
        

        
class Boosted_Regression():
    '''
    This class implements the regression model using XGBoost.

    The data comes from the DataPrep class and where it is split into training, validation and testing sets.
    '''
    def __init__(self, data, nucleus, level, TN):
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.X_just_test, self.y_just_test = data
        self.nucleus = nucleus #nucleus (12C or 16O)
        self.level = level #energy level 
        self.model = None #saved model
        #paths for high and low TN tables

        #CHANGE PATHS ACCORDINGLY !!!
        if TN == 'high':
            self.path = '/path/to/directory/boosted_model.bin'%(self.nucleus,self.level) #path for saved model
            self.rmse_path = '/path/to/directory/rmse.csv'%(self.nucleus,self.level) #path for learning curves
        if TN == 'low':
            self.path = '/path/to/directory/boosted_model.bin'%(self.nucleus,self.level) #path for saved model
            self.rmse_path = '/path/to/directory/rmse.csv'%(self.nucleus,self.level) #path for learning curves


    def train(self):
        '''
        This function trains the model using the training set.
        '''
        print('Training the model...')
        start = time.time()
        eval_set = [(self.X_train, self.y_train), (self.X_val, self.y_val)] #sets for RMSE evaluation of learning curves
        self.model = xgb.XGBRegressor(min_child_weight=4,gamma=0, max_depth=13, eta=0.1, n_estimators=200,
                                      tree_method='approx', eval_metric='rmse', objective='reg:pseudohubererror',huber_slope=10)
        self.model.fit(self.X_train, self.y_train, eval_set=eval_set, verbose=False)
        end = time.time()
        print('Finished training!')
        print('Training time: %g seconds'%(end-start))

        self.train_rmse = self.model.evals_result()['validation_0']['rmse'] #training RMSE evolution with n_estimators
        self.val_rmse = self.model.evals_result()['validation_1']['rmse'] #validation RMSE evolution with n_estimators
    
        rmse_df = pd.DataFrame({'training':self.train_rmse, 'validation':self.val_rmse}) #dataframe of RMSE learning curves
        #uncomment following line to save learning curves to self.rmse_path
        #rmse_df.to_csv(self.rmse_path, index=False)

    def save(self):
        '''
        This function saves the trained model. If the model has not been trained, it raises an error.
        '''
        if self.model:
            self.model.save_model(self.path)
        else:
            raise ValueError('TRAIN THE MODEL FIRST!!!')
        
    def load(self):
        '''
        This function loads the trained model.
        '''
        self.model = xgb.Booster()
        self.model.load_model(self.path)

    def hp_tuning(self,params):   
        '''
        This function tunes the hyperparameters of the model using a grid search.

        Args:
            params: hyper-parameters we want to search

        Returns:
            best_params: best hyper-parameters found by the grid search
            best_score: score achieved by the best hyper-parameters
        '''
        model = xgb.XGBRegressor()
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, verbose=3, scoring='neg_root_mean_squared_error')
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        return best_params, best_score
    
    def learning_curves(self):
        '''
        This function plots the learning curves of the RMSE for the training and validation sets
        '''
        lc_df = pd.read_csv(self.rmse_path)
        train_rmse = lc_df['training'] #training RMSE
        val_rmse = lc_df['validation'] #validation RMSE
        plt.figure()
        plt.plot(train_rmse, label = 'Training %g'%np.round(train_rmse.tail(10).mean(),5), color='green') 
        plt.plot(val_rmse, label='Validation %g'%np.round(val_rmse.tail(10).mean(),5), color='tab:orange', linestyle='--')
        plt.xlabel('Number of Trees')
        plt.ylabel('RMSE')
        plt.ylim(0,0.4)
        plt.legend(frameon=False) #the labels are the average over the last 10 entries in the RMSEs 
        plt.show()
        
    def evaluation(self, set):
        '''
        This function evaluates the model on any of the sets (training, validation, testing or just_test).

        Args:
            set: A string that specifies the set to be evaluated (train/test/val/just_test).

        Returns:
            pred_df: A dataframe containing the real and imaginary parts of each component of the tensor predicted by the model.
            rmse_Re: A dictionary containing the RMSE of the real part of each component of the tensor.
            rmse_Im: A dictionary containing the RMSE of the imaginary part of each component of the tensor.
            r2_Re: A dictionary containing the R2 of the real part of each component of the tensor.
            r2_im: A dictionary containing the R2 of the imaginary part of each component of the tensor.
        '''
        #choosing the set to be evaluated acccording to the input
        sets = {'train':(self.X_train, self.y_train), 'test': (self.X_test, self.y_test), 'val': (self.X_val, self.y_val), 'just_test': (self.X_just_test,self.y_just_test)}
        self.X_set, self.y_set = sets[set]
        #predicting the values of the set
        self.preds = self.model.predict(xgb.DMatrix(self.X_set))
        self.pred_df = pd.DataFrame(data = self.preds, columns = self.y_set.columns)

        #combining all the testing/validation into ito one dataframe
        self.all = pd.concat([self.X_set.reset_index(drop=True),self.y_set], axis=1)

        #empty dictionaries
        rmse = {}
        r2 = {}

        #iterating over the tensor components
        for i,j in enumerate(self.y_set):
            #if all the predicted values in a column are nearly zero(less than 1e-13), set them to zero
            if self.pred_df[j].lt(1e-4).all():
                self.pred_df[j] = 0
            rmse[i] = np.sqrt(mean_squared_error(self.y_set[j], self.pred_df[j])) #calculating the RMSE for each component
            r2[i] = r2_score(self.y_set[j], self.pred_df[j]) #calculating the R2 score for each component

        #splitting the RMSE and R2 score into the real and imaginary parts
        self.rmse_Re = dict(list(rmse.items())[0::2])
        self.rmse_Im = dict(list(rmse.items())[1::2])
        self.r2_Re = dict(list(r2.items())[0::2])
        self.r2_Im = dict(list(r2.items())[1::2])
        return self.pred_df,self.rmse_Re, self.rmse_Im, self.r2_Re, self.r2_Im
    
    def single_event(self,T,θ,p3,E):
        '''
        This function predicts the tensor component for a single event (one set of kinematics)

        Args:
            T,θ,p3,E: kinematics of single event

        Returns:
            single_pred: tensor component values
        '''
        feat = pd.DataFrame(data = [[T,θ,p3,E]], columns = self.X_train.columns)
        single_pred = self.model.predict(xgb.DMatrix(feat))
        return single_pred
        
    def plot_rmse(self):
        '''
        Ths function plot the RMSE of the real and imaginary parts for each component of the tensor in the form of a bar plot.
        '''
        plt.bar(range(len(self.rmse_Re)), list(self.rmse_Re.values()), align='center', label='Real', alpha=0.8, width = 0.5, color='red')
        plt.bar(range(len(self.rmse_Im)), list(self.rmse_Im.values()), align='edge', label='Imaginary', alpha=0.8, width = 0.5, color='blue')
        plt.xticks(range(10), [0,1,2,3,4,5,6,7,8,9])
        plt.xlabel('Tensor Component')
        plt.ylabel('RMSE')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_r2(self):
        '''
        Ths function plot the R2 score of the real and imaginary parts for each component of the tensor in the form of a bar plot.
        '''
        plt.bar(range(len(self.r2_Re)), list(self.r2_Re.values()), align='center', label='Real', alpha=0.8, width = 0.5, color='red')
        plt.bar(range(len(self.r2_Im)), list(self.r2_Im.values()), align='edge', label='Imaginary', alpha=0.8, width = 0.5, color='blue')
        plt.xticks(range(10), [0,1,2,3,4,5,6,7,8,9])
        plt.xlabel('Tensor Component')
        plt.ylabel('R2 Score')
        plt.grid()
        plt.ylim(0.98,1)
        plt.legend()
        plt.show()

    def plot_predictions(self,x):
        '''
        This function plots the real and imaginary parts of each component of the tensor against one of the kinematic variables while keeping the others constant.
        It plots the true table values and the predicted values.

        Args:
            x: A string that specifies the kinematic variable that is varied on the x-axis.
        '''
        #dictionary that contains the other kinematic variables for each variable
        var = {'T':['θ','p3','E'], 'θ':['T','p3','E'], 'p3':['θ','T','E'], 'E':['θ','p3','T']}
        #function that returns the most common value for a given variable
        most = lambda k: self.all[k].value_counts().idxmax()
        #a dataframe where the only varied kinematic is the one specified by the input and the others are kept constant according to their most common value
        a = self.all.loc[(self.all[var[x][0]] == most(var[x][0])) & (self.all[var[x][1]] == most(var[x][1])) & (self.all[var[x][2]] == most(var[x][2]))]
        #a dataframe that contains the predicted values for the same kinematic variables as the dataframe a
        p = self.pred_df[self.pred_df.index.isin(a.index)]
        p = pd.concat([a[['T','θ','p3','E']],p], axis=1)
  
        #one plot for the real parts of the components
        plt.figure(figsize=(15,6))
        plt.suptitle('Constants: %s = %g, %s = %g, %s = %g'%(var[x][0],most(var[x][0]),var[x][1],most(var[x][1]),var[x][2],most(var[x][2])))
        for n in range(10):
            plt.subplot(2,5,n+1)
            plt.plot(p.sort_values(by=[x])[x],p.sort_values(by=[x])['Re%i'%n], color='black', label='ML') #ML predictions
            plt.scatter(a.sort_values(by=[x])[x],a.sort_values(by=[x])['Re%i'%n], marker='+',color='red',alpha=0.8,label='Table') #true table values
            plt.xlabel(x)
            plt.ylabel('Re$\{H^{\\mu\\nu}[%i]\}$'%n)
            plt.grid()
            plt.tight_layout()
        plt.legend()
        plt.show()

        #one plot for the imaginary parts of the components
        plt.figure(figsize=(15,6))
        for n in range(10):
            plt.subplot(2,5,n+1)
            plt.plot(p.sort_values(by=[x])[x],p.sort_values(by=[x])['Im%i'%n], color='black', label='ML') #ML predictions
            plt.scatter(a.sort_values(by=[x])[x],a.sort_values(by=[x])['Im%i'%n], marker='+',color='b',alpha=0.8,label='Table') #true table values
            plt.xlabel(x)
            plt.ylabel('Im$\{H^{\\mu\\nu}[%i]\}$'%n)
            plt.grid()
            plt.tight_layout()
        plt.legend()
        plt.show()

    def errors(self,x,y,z):
        '''
        This function plots the absolute error between the true table values and the predicted values for each component on a 2D plot of the phase space.
        The color of the points represents the absolute error.

        Args:
            x: A string that specifies the kinematic variable that is on the x-axis.
            y: A string that specifies the kinematic variable that is on the y-axis.
            z: A string that specifies the part of the tensor (Re or Im) to be plotted.
        '''

        kin = {'T', 'θ', 'p3', 'E'}
        axes = set([x,y]) #the kinematics in the x and y axes
        const = list(kin-axes) #the kinematics that are kept constant
        err = self.y_set.reset_index(drop=True).sub(self.pred_df).abs() #calculating the absolute errors
        self.err_df = pd.concat([self.X_set.reset_index(drop=True), err], axis=1) #combining the errors with the corresponding kinematics
        const1 = self.err_df[const[0]].value_counts().idxmax() #finds the most common value of the first constant kinematic (can be changed here)
        const2 = self.err_df[const[1]].value_counts().idxmax() #finds the most common value of the second constant kinematic (can be changed here)
        err_plot = self.err_df.loc[(self.err_df[const[0]] == const1) & (self.err_df[const[1]] == const2)] #a dataframe containing the values that will be plotted
        cm = 'coolwarm'
        cbar = plt.cm.ScalarMappable(cmap=cm)
        cbar.set_array([])
        z_min, z_max = float('inf'), float('-inf')
        plt.figure(figsize=(15,6))
        for i in range(10):
            plt.subplot(2,5,i+1)
            X,Y,Z = err_plot[x], err_plot[y], err_plot['%s%i'%(z,i)]
            z_min = min(z_min, Z.min())
            z_max = max(z_max, Z.max())
            plt.suptitle('Constants: %s = %g, %s = %g'%(const[0],const1,const[1],const2))
            plt.scatter(X,Y, c=Z, cmap=cm, marker='s', s = 20) #change marker size (s) to make plots look continuous according the set size
            plt.title('Component %i'%i)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.tight_layout()

        cbar.set_clim(z_min, z_max)
        plt.subplots_adjust(right=0.9)
        cbar_ax = plt.gcf().add_axes([0.92, 0.09, 0.02, 0.8])
        plt.colorbar(cbar, cax=cbar_ax, label = 'Absolute Error')
        plt.show()

    def residuals(self, comp, sigma):
        '''
        A function that creates various types of plots of the residuals.
        It also calculates the percentage of outliers that are predicted by the model.

        Args:
            comp: A string that specifies the part of the tensor (Re or Im) to be plotted in certain figures.
            sigma: cut-off confidence region beyond which predictions are outliers (2 or 3)

        Returns:
            outliers_per_Re: the percentage of outliers in the real parts of the components
            outliers_per_Im: the percentage of outliers in the imgainry parts of the components
            all_out: the percentage of outliers for all of the targets
        '''
        resid = self.y_set.reset_index(drop=True).sub(self.pred_df) #calculating the residuals
        std_res = resid/resid.std(ddof=1) #calculating the standardised residuals
       
        #THE DENSITY PLOT OF THE STANDARDISED RESIDUALS    
        plt.figure(figsize=(7, 5))
        plt.grid(zorder=1)
        sns.kdeplot(std_res['Re1'], fill=True, bw_adjust=1.5, zorder=2) #change tensor component you wish to plot eg. Re1, Im5, Re8, ...
        plt.axvline(-3, color='red', linestyle='--', label='$\pm3 \sigma$') #3σ region
        plt.axvline(3, color='red', linestyle='--') #3σ region
        plt.axvline(-2, color='black', linestyle='--', label='$\pm2 \sigma$') #2σ region
        plt.axvline(2, color='black', linestyle='--') #2σ region
        plt.xlabel('Standardised Residuals')
        plt.ylabel('Density')
        plt.legend()
        plt.xlim(-5,5)
        plt.show()
        
        #HISTOGRAMS OF THE RESIDUALS FOR EACH TENSOR COMPONENT
        #comp: Re or Im
        plt.figure(figsize=(17,7))
        for i in range(10):
            plt.subplot(2,5,i+1)
            plt.suptitle('Distribution of Residuals (%s)'%comp)
            plt.hist(resid['%s%i'%(comp,i)], bins=100, color='r', label = '$\mu=%g$ \n $\sigma=%g$' %(np.round(resid['%s%i'%(comp,i)].mean(),3),np.round(resid['%s%i'%(comp,i)].std(),3)))
            plt.xlim(resid['%s%i'%(comp,i)].min()/5,resid['%s%i'%(comp,i)].max()/5)
            plt.title('Component %i'%i)
            plt.xlabel('Residuals')
            plt.ylabel('Counts')
            plt.tight_layout()
            plt.legend(loc = 'upper right')
            plt.grid()
        plt.show()
        
        #SCATTER PLOTS OF THE TRUE AND PREDICTED VALUES FOR ALL COMPONENTS
        plt.figure(figsize=(15,6))
        for i in range(10):
            plt.subplot(2,5,i+1)
            plt.scatter(self.y_set['%s%i'%('Re',i)],self.pred_df['%s%i'%('Re',i)], alpha=0.5, color='coral', marker='.', label='Real') #real parts
            plt.scatter(self.y_set['%s%i'%('Im',i)],self.pred_df['%s%i'%('Im',i)], alpha=0.5, color='skyblue', marker='.', label='Imaginary') #imaginary parts
            plt.axline([0,0],[1,1], color='black', ls='--', alpha = 0.7, label='Identity') #y=x diagonal
            plt.title('Component %i'%i)
            plt.xlabel('True Value')
            plt.ylabel('Predicted Value')
            plt.legend(frameon=False)
            plt.tight_layout()
        plt.show()
       
        #SCATTER PLOT OF THE STANDARDISED RESIDUALS WRT KINEMATICS OR PREDICTED VALUES
        plt.figure(figsize=(15,6))
        #empty lists for outliers per component
        outliers_per_Re = []
        outliers_per_Im = []
        for i in range(10):
            plt.subplot(2,5,1+i)
            x = self.X_set['T'] #kinematic on x axis
            plt.scatter(x,std_res['%s%i'%('Re',i)], alpha=0.2, color='red', marker='.', label='Real') #change x to self.pred_df['%s%i'%('Re',i)] for plot wrt to predictions
            plt.scatter(x,std_res['%s%i'%('Im',i)], alpha=0.2, color='blue', marker='.', label='Imaginary') #change x to self.pred_df['%s%i'%('Im',i)] for plot wrt to predictions
            plt.axhline(y=-2, color='black', alpha=0.4, ls = '--') #2σ region
            plt.axhline(y=2, color='black', alpha=0.4, ls = '--') #2σ region
            plt.axhline(y=3, color='black') #3σ region
            plt.axhline(y=-3, color='black') #3σ region
            #plt.ylim(-25,25)
            plt.title('Component %i'%i)
            plt.xlabel('$T_N$') #change according to variable on x axis
            plt.ylabel('$\\Delta_{std}$')
            plt.legend(frameon=False, loc='upper right')
            plt.tight_layout()

            #calculates percentage of outliers per tensor component
            #outliers are the predicted values with standardised residuals larger than the specified sigma value (2 or 3)
            out_Re = (std_res['%s%i'%('Re',i)].abs() > sigma).sum()/len(std_res)*100 
            out_Im = (std_res['%s%i'%('Im',i)].abs() > sigma).sum()/len(std_res)*100
            outliers_per_Re.append(out_Re)
            outliers_per_Im.append(out_Im)
        plt.show()
        all_out = outliers_per_Re+outliers_per_Im
        
        #BAR PLOT OF THE PERCENTAGE OF OUTLIERS FOR THE REAL AND IMAGINARY PARTS OF ALL COMPONENTS
        plt.figure(figsize=(10,6))
        plt.grid(zorder=1)
        plt.bar(range(len(outliers_per_Re)), outliers_per_Re, align='center', alpha=0.8, color='purple', label='Real', width=0.4, zorder=2)
        plt.bar(range(len(outliers_per_Im)), outliers_per_Im, align='edge', alpha=0.8, color='orange', label='Imaginary', width=0.4, zorder=2)
        plt.xticks(range(10), [0,1,2,3,4,5,6,7,8,9])
        plt.xlabel('Tensor Component')
        plt.ylabel('Percentage (%)')
        plt.title('Standarised Residuals Outside %g$\sigma$'%sigma)
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.show()
        return outliers_per_Re, outliers_per_Im, all_out
        
        
############################################################################################################     
           

#SPECIFYING TABLE
#Example
level = 7 
nucleus = 'Oxygen'
TN = 'high'

#columns in tables
colnames = ['T','θ','p3','E','ij','Re','Im'] 


#load high or low TN table, change path accordingly
if TN == 'high':
    table = pd.read_csv('/path/to/directory/%s/table%i/Hmunu.out'%(nucleus,level),names=colnames, delim_whitespace=True, header=None)

if TN == 'low':
    table = pd.read_csv('/path/to/directory/%s/lowTN/table%i/Hmunu.out'%(nucleus,level),names=colnames, delim_whitespace=True, header=None)


#DATA PRE-PROCESSING AND PREPARATION
    
#removing outliers from level 0 tables
if level == 0:
    real = DataPrep(table).sets().all_Re
    #jumps larger than 8000 are removed
    re_out = real[real['Re0'].diff().abs()>8000][['T','θ','p3','E']]
    ij = [0,3,9] #components where outliers were located
    reps = pd.concat([re_out.copy() for _ in ij], ignore_index=True)
    reps['ij'] = ij*len(re_out)
    match = pd.merge(reps, table, on=['T','θ','p3','E','ij'], how='inner')
    table = table[~table.set_index(['T','θ','p3','E','ij']).index.isin(match.set_index(['T','θ','p3','E','ij']).index)]


#PREPARING THE TRAINING, TESTING AND VALIDATION SETS
data = DataPrep(table).sets().split(scaling=True)

#MODEL IMPLEMENTATION
model = Boosted_Regression(data,nucleus,level,TN)

#UNCOMMENT THE FOLLOWING SAMPLE COMMANDS TO RUN SCRIPT
#REMEMBER TO CHANGE PATHS FOR LOADING TABLES AND STORING MODELS

#model.train()
#model.save()
#model.load()
#model.learning_curves()
#model.evaluation('test') #this method must be called before the visualisation ones (plot_rmse(),plot_r2(),plot_predictions(),errors(), residuals())
#model.plot_rmse()
#model.plot_r2()
#model.plot_predictions('θ')
#model.errors('p3','θ','Re')
#model.residuals('Re',2)