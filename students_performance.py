'''
No GPA as a feature.
'''
from enum import Enum
from  matplotlib import pyplot as plt
import numpy as np
import os
from typing import Dict, Tuple, List, Union

import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


data_path = "/Users/elin/Documents/University/CMU/2025_Spring/applied_ML/students_performance/"
data_file = "Student_performance_data.csv"
data_full_path = os.path.join(data_path, data_file)
# StudentID, Age, Gender, Ethnicity, ParentalEducation,
# StudyTimeWeekly, Absences, Tutoring, ParentalSupport, Extracurricular,
# Sports, Music, Volunteering, GPA, GradeClass

class Column(Enum):
    StudentID = 0
    Age = 1
    Gender = 2
    Ethnicity = 3
    ParentalEducation = 4
    StudyTimeWeekly = 5
    Absences = 6
    Tutoring = 7
    ParentalSupport = 8
    Extracurricular = 9
    Sports = 10
    Music = 11
    Volunteering = 12
    GPA = 13
    GradeClass = 14


class ImgHandler():
    def __init__(self, data_frame: pd.DataFrame) -> None:
        self.__df__: pd.DataFrame = data_frame

    def __plot_ending__(self, label_name: Tuple[str, str], title_name: str, img_name: str):
        plt.xlabel(label_name[0])
        plt.ylabel(label_name[1])
        plt.title(title_name)
        plt.savefig(os.path.join(data_path, 'img/' + img_name + '.jpg'))

    # ==================== histogram ====================

    def plot_histogram(self, data, label_name: Tuple[str, str],    title_name: str, img_name: str, x_range:Tuple[float]=None) -> None:
        self.__plot_histogram__(data, label_name, title_name, img_name, x_range)

    def __plot_histogram__(self, data, label_name: Tuple[str, str], title_name: str, img_name: str, x_range:Tuple[float]=None) -> None:
        if x_range:
            plt.hist(data, edgecolor='black', range=x_range)
        else:
            plt.hist(data, edgecolor='black')
        self.__plot_ending__(label_name, title_name, img_name)

    def plot_study_time_histogram(self) -> None:
        # print(type(Column.StudyTimeWeekly.value))
        study_time_col = self.__df__.columns[Column.StudyTimeWeekly.value]
        # print(type(self.__df__[study_time_col]))
        hist_data = self.__df__[study_time_col]
        label_name = ('Study Time (hr/week)',
                      'Number of Students')
        title = 'Weekly Study Time Distribution of Students'
        img_name = 'weekly_study_time_distribution_of_students'
        self.__plot_histogram__(hist_data, label_name, title, img_name)

    def plot_gender_histogram(self) -> None:
        gender_col = self.__df__.columns[Column.Gender.value]
        hist_data = self.__df__[gender_col]
        label_name = ('Gender',
                      'Number of Students')
        title = 'Number of Student in Each Gender'
        img_name = 'gender_histogram'
        plt.xticks([0, 1], ['Male', 'Female'])
        self.__plot_histogram__(hist_data, label_name, title, img_name, x_range=(-0.5, 1.5))

    # ==================== scatter ====================

    def plot_scatter(self, data_x_y: Tuple, label_name: Tuple[str, str], title_name: str, img_name: str) -> None:
        self.__plot_scatter__(data_x_y, label_name, title_name, img_name)

    def __plot_scatter__(self, data_x_y: Tuple, label_name: Tuple[str, str], title_name: str, img_name: str) -> None:
        plt.scatter(data_x_y[0], data_x_y[1], s=5)
        self.__plot_ending__(label_name, title_name, img_name)

    def plot_study_time_gpa_scatter(self) -> None:
        study_time_col = self.__df__.columns[Column.StudyTimeWeekly.value]
        gpa_class_col = self.__df__.columns[Column.GPA.value]
        data_x = self.__df__[study_time_col]
        data_y = self.__df__[gpa_class_col]
        label_name = ('Study Time (hr/week)',
                      'GPA')
        title = 'Weekly Study Time vs. GPA'
        img_name = 'weekly_study_time_vs_gpa'
        self.__plot_scatter__((data_x, data_y), label_name, title, img_name)

    # ==================== heat map ====================

    def __plot_heat_map__(self, data, label_name: Tuple[str, str], title_name: str, img_name: str, ticks: Tuple[List[int]]) -> None:
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.xticks(ticks=np.arange(len(ticks[0]))+ 0.5, labels=ticks[0])
        plt.yticks(ticks=np.arange(len(ticks[1]))+ 0.5, labels=ticks[1])
        self.__plot_ending__(label_name, title_name, img_name)

    def plot_parent_education_vs_support_heat(self) -> None:
        education_col = self.__df__.columns[Column.ParentalEducation.value]
        support_col = self.__df__.columns[Column.ParentalSupport.value]
        # data = self.__df__[[education_col, support_col]]
        
        heatmap_data = pd.crosstab(self.__df__[education_col], self.__df__[support_col])

        label_name = ['Education', 'Support']
        title_name = "Parential Education vs Parential Support"
        img_name = "Education vs Support"
        self.__plot_heat_map__(heatmap_data, label_name, title_name, img_name)
        
    def plot_parent_support_vs_gpa_heat(self) -> None:
        support_col = self.__df__.columns[Column.ParentalSupport.value]
        grade_class_col = self.__df__.columns[Column.GradeClass.value]
        pd.set_option('display.max_rows', None)
        heatmap_data = pd.crosstab(self.__df__[grade_class_col], self.__df__[support_col])

        label_name = ['Grade_Class', 'Parental_Support']
        title_name = "Grade Class vs Parental Support"
        img_name = "Grade_vs_Parental_Support"
        # plt.xticks([0, 1, 2, 3, 4], ['A', 'B', 'C', 'D', 'E'])
        # plt.yticks([0, 1, 2, 3, 4], ['None', 'Low', 'Moderate', 'High', 'Very High'])
        xticks = ['A', 'B', 'C', 'D', 'E']
        yticks = ['None', 'Low', 'Moderate', 'High', 'Very High']
        ticks = (xticks, yticks)
        self.__plot_heat_map__(heatmap_data, label_name, title_name, img_name, ticks)

    # ==================== box plot ====================

    def __plot_boxplot__(self, data, label_name: Tuple[str], title_name: str, img_name: str, scatter=False) -> None:
        plt.boxplot(data, tick_labels=label_name[0], showmeans=True)
        if scatter:
            for i, dataset in enumerate(data, start=1): 
                x = np.random.normal(i, 0.08, size=len(dataset))  # add jitter
                plt.scatter(x, dataset, alpha=0.1, color="blue", s=15)
        plt.ylabel(label_name[1])
        plt.title(title_name)
        plt.savefig(os.path.join(data_path, 'img/' + img_name + '.jpg'))
        # self.__plot_ending__(label_name, title_name, img_name)

    def plot_study_time_weekly_w_age(self) -> None:
        age_study_time_dict: Dict[int, List[float]] = {}
        for data in self.__df__.itertuples():
            # print(data.StudyTimeWeekly)
            if age_study_time_dict.get(data.Age):
                age_study_time_dict[data.Age].append(data.StudyTimeWeekly)
            else:
                age_study_time_dict[data.Age] = [data.StudyTimeWeekly]
        # print(age_study_time_dict)
        x_label = sorted(age_study_time_dict.keys())
        box_plot_data = [age_study_time_dict[label] for label in x_label]
        y_label = "Study Time Weekly (Hours)"
        label_name = [x_label, y_label]
        title_name = "Study Time Weekly in Different Age"
        # img_name = "study_time_in_different_age-normal"
        img_name = "study_time_in_different_age-scatter"
        self.__plot_boxplot__(box_plot_data, label_name, title_name, img_name, True)

    def plot_absence_w_grade_class(self) -> None:
        grade_absence_dict: Dict[int, List[float]] = {}
        for data in self.__df__.itertuples():
            # print(data.StudyTimeWeekly)
            if grade_absence_dict.get(data.GradeClass):
                grade_absence_dict[data.GradeClass].append(data.Absences)
            else:
                grade_absence_dict[data.GradeClass] = [data.Absences]
        x_label = sorted(grade_absence_dict.keys())
        box_plot_data = [grade_absence_dict[label] for label in x_label]
        y_label = "Absences (Times)"
        label_name = [x_label, y_label]
        title_name = "Absences and Grade Class"
        # img_name = "absences_and_grade-normal"
        # self.__plot_boxplot__(box_plot_data, label_name, title_name, img_name, False)
        img_name = "absences_and_grade-scatter"
        self.__plot_boxplot__(box_plot_data, label_name, title_name, img_name, True)


class DataHandler():
    def __init__(self) -> None:
        self.__df__: pd.DataFrame = None
        self.__img_handler__: ImgHandler = None

        self.__read_csv__()
        self.__remove_wrong_data__()
        self.__set_img_handler__()

    def __read_csv__(self) -> None:
        self.__df__ = pd.read_csv(data_full_path)
        # print(self.__df__)
        # print(self.__df__.iloc[0]) # print the first row
        # print(self.__df__.columns) # print column name
        # print(self.__df__[self.__df__.columns[0]]) # print the first column

    def __set_img_handler__(self) -> None:
        self.__img_handler__ = ImgHandler(self.__df__)

    # ====================  ====================

    def check_gpa_grade(self) -> None:
        '''Check the wrong data about the relationship between GPA and GradeClass'''
        gpa_col = self.__df__.columns[13]
        grade_class_col = self.__df__.columns[14]
        print(gpa_col, grade_class_col)
        all_bad: Dict[int, Tuple[int]] = {}
        for i, gpa in enumerate(self.__df__[gpa_col]):
            grade_class = self.__df__[grade_class_col][i]
            if gpa >= 3.5:
                if grade_class != 0:
                    all_bad[i] = (gpa, grade_class)
                    print(i, gpa, grade_class)
            elif 3.5 > gpa and gpa >= 3.0:
                if grade_class != 1:
                    all_bad[i] = (gpa, grade_class)
                    print(i, gpa, grade_class)
            elif 3.0 > gpa and gpa >= 2.5:
                if grade_class != 2:
                    all_bad[i] = (gpa, grade_class)
                    print(i, gpa, grade_class)
            elif 2.5 > gpa and gpa >= 2.0:
                if grade_class != 3:
                    all_bad[i] = (gpa, grade_class)
                    print(i, gpa, grade_class)
            else:
                if grade_class != 4:
                    all_bad[i] = (gpa, grade_class)
                    print(i, gpa, grade_class)
        print(len(all_bad))

    def __remove_wrong_data__(self) -> None:
        gpa_col = self.__df__.columns[13]
        grade_class_col = self.__df__.columns[14]
        row_num = self.__df__.shape[0]
        for row in range(row_num):
            grade_class = self.__df__[grade_class_col][row]
            gpa = self.__df__[gpa_col][row]
            # print(row, gpa, grade_class)
            if gpa >= 3.5:
                if grade_class != 0:
                    self.__df__ = self.__df__.drop(index=row)
            elif 3.5 > gpa and gpa >= 3.0:
                if grade_class != 1:
                    self.__df__ = self.__df__.drop(index=row)
            elif 3.0 > gpa and gpa >= 2.5:
                if grade_class != 2:
                    self.__df__ = self.__df__.drop(index=row)
            elif 2.5 > gpa and gpa >= 2.0:
                if grade_class != 3:
                    self.__df__ = self.__df__.drop(index=row)
            else:
                if grade_class != 4:
                    self.__df__ = self.__df__.drop(index=row)
        # print(self.__df__)

    # ==================== linear regression ====================

    def __get_features__(self) -> pd.DataFrame:
        ignore_col = [self.__df__.columns[Column.StudentID.value],
                      self.__df__.columns[Column.Age.value],
                      self.__df__.columns[Column.Gender.value],
                      self.__df__.columns[Column.Ethnicity.value],
                      self.__df__.columns[Column.Sports.value],
                      self.__df__.columns[Column.Music.value],
                      self.__df__.columns[Column.Volunteering.value],
                      self.__df__.columns[Column.GPA.value],
                      self.__df__.columns[Column.GradeClass.value]]
        return self.__df__.drop(columns=ignore_col)

    def __get_response_var__(self) -> pd.DataFrame:
        gpa_class_col = self.__df__.columns[Column.GPA.value]
        return self.__df__[gpa_class_col]

    def __plot_predict_vs_observe__(self, pred, obs, r2) -> None:
        label_name = ["Predicted GPA", "Observed GPA"]
        title_name = "Predicted GPA vs Observed GPA"
        img_name = "Predicted_vs_Observed"
        r2_str = f"R^2 = {r2:.2f}"
        __img_handler = ImgHandler(None)
        plt.text(0.2, 3.5, r2_str, fontsize=12, color='black')
        __img_handler.plot_scatter((pred, obs), label_name, title_name, img_name)

    def fit_linear_regression(self) -> None:
        x = self.__get_features__()
        y = self.__get_response_var__()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        linear_regression_model = LinearRegression()
        linear_regression_model.fit(x_train, y_train)
        y_pred = linear_regression_model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        print(f"R2 Score: {r2:.2f}")

        self.__plot_predict_vs_observe__(y_pred, y_test, r2)

    # ====================  ====================

    def fit_random_forest(self) -> None:
        x: pd.DataFrame = self.__get_features__()
        y: pd.DataFrame = self.__get_response_var__()
        trainer = Trainer(x, y)
        trainer.do_random_forest()

    def fit_decision_tree(self) -> None:
        x: pd.DataFrame = self.__get_features__()
        y: pd.DataFrame = self.__get_response_var__()
        trainer = Trainer(x, y)
        trainer.do_decision_tree()

    # ====================  ====================

    def __get_features_2__(self) -> pd.DataFrame:
        ignore_col = [self.__df__.columns[Column.StudentID.value],
                      self.__df__.columns[Column.GPA.value],
                      self.__df__.columns[Column.GradeClass.value]]
        return self.__df__.drop(columns=ignore_col)

    def __plot_complexity_vs_r2__(self, comps: List, r2_scores: List[float]) -> None:
        label_name = ["Model Complexity (Number of Features)",
                      "R2 Score"]
        title_name = "Model Complexity vs. R2 Score"
        img_name = "Complexity_vs_R2"
        __img_handler = ImgHandler(None)
        __img_handler.plot_scatter((comps, r2_scores), label_name, title_name, img_name)

    def fit_model_iterative(self) -> None:
        x = self.__get_features_2__()
        y = self.__get_response_var__()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        features = x.columns
        r2_scores = []
        complexities = []
        for i in range(1, len(features) + 1):
            selected_features = features[:i] 
            model = LinearRegression()
            model.fit(x_train[selected_features], y_train) 
            y_pred = model.predict(x_test[selected_features])
            r2 = r2_score(y_test, y_pred)
            r2_scores.append(r2)
            complexities.append(i)
        self.__plot_complexity_vs_r2__(complexities, r2_scores)

    # ==================== plot ====================

    def plot_study_time_histogram(self):
        self.__img_handler__.plot_study_time_histogram()

    def plot_gender_histogram(self):
        self.__img_handler__.plot_gender_histogram()

    def plot_study_time_gpa_scatter(self) -> None:
        self.__img_handler__.plot_study_time_gpa_scatter()

    def plot_parent_education_vs_support_heat(self) -> None:
        self.__img_handler__.plot_parent_education_vs_support_heat() 

    def plot_parent_support_vs_gpa_heat(self) -> None:
        self.__img_handler__.plot_parent_support_vs_gpa_heat()

    def plot_study_time_weekly_w_age(self) -> None:
        self.__img_handler__.plot_study_time_weekly_w_age()

    def plot_absence_w_grade_class(self) -> None:
        self.__img_handler__.plot_absence_w_grade_class()

    def standardize_age(self) -> None:
        scaler = StandardScaler()
        age_col = self.__df__.columns[Column.Age.value]
        # print(self.__df__[age_col])
        scaled_data = scaler.fit_transform(self.__df__[[age_col]])
        print(scaled_data[0])
        label_name = ('Age',
                      'Number of Students')
        title = 'Number of Student in Each Age'
        img_name = 'age_histogram'
        # plt.xticks([0, 1], ['Male', 'Female'])
        self.__img_handler__.plot_histogram(scaled_data, label_name, title, img_name)

    def plot_img(self) -> None:
        # self.plot_gender_histogram()
        # self.plot_study_time_histogram()
        # self.plot_study_time_gpa_scatter()
        # self.plot_parent_education_vs_support_heat()
        # self.plot_parent_support_vs_gpa_heat()
        # self.plot_study_time_weekly_w_age()
        # self.plot_absence_w_grade_class()
        self.standardize_age()


class Trainer():
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.__data__: List[pd.DataFrame] = [x, y]
        self.__train__: List[pd.DataFrame] = None
        self.__validate__: List[pd.DataFrame] = None
        self.__test__: List[pd.DataFrame] = None
        self.__split_data__()

    def __split_data__(self) -> None:
        x_train, x_validate_test, y_train, y_validate_test = train_test_split(self.__data__[0], self.__data__[1], test_size=0.2, random_state=1)

        x_validate, x_test, y_validate, y_test = train_test_split(x_validate_test, y_validate_test, test_size=0.5, random_state=1)
        self.__train__ = [x_train, y_train]
        self.__validate__ = [x_validate, y_validate]
        self.__test__ = [x_test, y_test] 

    def __check_result__(self, real_y, predict_y, item:str="") -> None:
        mse = mean_squared_error(real_y, predict_y)
        mae = mean_absolute_error(real_y, predict_y)
        r2 = r2_score(real_y, predict_y)
        print(f"{item} MSE: {mse:.2f}")
        print(f"{item} MAE: {mae:.2f}")
        print(f"{item} R² Score: {r2:.2f}")

    def __check_validate_result__(self, y_val_predict) -> None:
        self.__check_result__(self.__validate__[1], y_val_predict, "Validation")

    def __check_test_result__(self, y_test_predict) -> None:
        self.__check_result__(self.__test__[1], y_test_predict, "Test")

    def __fit_model__(self, model: Union[DecisionTreeRegressor, RandomForestRegressor, KMeans]) -> None:
        model.fit(self.__train__[0], self.__train__[1])
        y_val_predict = model.predict(self.__validate__[0])
        self.__check_validate_result__(y_val_predict)
        y_test_predict = model.predict(self.__test__[0])
        self.__check_test_result__(y_test_predict)

    def do_random_forest(self) -> None:
        random_forest = RandomForestRegressor(n_estimators=100, random_state=1)
        self.__fit_model__(random_forest)

    def do_decision_tree(self) -> None:
        decision_tree = DecisionTreeRegressor()
        self.__fit_model__(decision_tree)



if __name__ == "__main__":
    data_handler = DataHandler()
    # data_handler.check_gpa_grade()
    # data_handler.plot_img()
    # data_handler.fit_linear_regression()
    # data_handler.fit_model_iterative()
    # data_handler.fit_random_forest()
    data_handler.fit_decision_tree()