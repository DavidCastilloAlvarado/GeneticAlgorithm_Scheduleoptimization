#%%
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import matplotlib as plt
import cv2
# %%
def crear_poblacion(cant_poblacion, cant_horas,cant_dias,cant_idiomas,cant_secc,cant_sesiones):
    cant_genes       = cant_idiomas*cant_secc*cant_sesiones  
    cant_bin_per_gen = 0
    for num in [cant_horas*cant_dias]:
        cant_bin_per_gen += len(str(bin(num))[2:]) 
    temp_pobl = np.random.random_integers(low= 0, high=1,size=(cant_idiomas,cant_bin_per_gen*cant_genes) )
    return temp_pobl,cant_bin_per_gen

def eval_poblac_fitness(poblac_init,cant_horas,cant_dias,cant_idiomas,cant_secc,cant_sesiones,cant_bin_per_gen):
    """
    Se han reconocido 3 tipos de cruces de horarios
    1) Cruces entre secciones - mismo curso        = x1000
    2) Cruces entre secciones - diferenctes curso  = x1
    3) Cruces entre sesiones del mismo curso       = x100
    Función de evaluación:
    f(x1 + x2 + x3) = -(x1*1000+x2+x3*100)
    ## EVALUACIÓN
    1 - comprobar traslape entre seSSiones del mismo curso    =>> verificar que los casilleros de las seSSiones sean diferentes | Dentro de la misma sección A o B de cada curso, comprobar que las dos sesiones dentro de la sección A o B sean diferentes.
        EJEMP: La sesión1 de la semana se traslapan con la sesión2 de la semana
    2 - comprobar traslape entre seCCiones del mismo curso    =>> verificar que los casilleros de las seSSiones sean diferentes | Agrupando cada seCCión A y B para un mismo curso, comprobar similitud de casillero de sesión entre A y B del mismo curso
        EJEMP: La sección A y B del español se traslapan
    3 - comprobar traslape entre seCCiones de DIFERENTE curso =>> verifivar que los casilleros de las seSSiones sean diferentes | Agrupando cada curso con su sección A y B, para compararla con la de otro
        EJEMP: La sección AóB del español se traslapa con la sección AóB del Ingles en alguna sesión de la semana
    """


    return 0

def seleccion():
    return 0

def crossover():
    return 0

def mutacion():
    return 'mutado'

def print_eval_result():
    return 0

def print_producto():
    return 0

#%%
if __name__ == "__main__":
    max_iteraciones = 1000  ## Cantidad máxima de iteraciones si no llega a cumplir ningun criterio
    cant_poblacion  = 20    ## Cantidad de individuos por población
    cant_horas      = 2     #['19-21','21-23' ] ## Cada 2 horas sin traslape de horas
    cant_dias       = 6     #['L','Mt','Mr','J','V','S']## Días disponibles para dar clases a la semana
    cant_idiomas    = 10    ## Cantidad de idiomas
    cant_secc       = 2     ## canridad de secciones por curso
    cant_sesiones   = 2     ## Cantidad de sesiones por semana
    cant_genes      = cant_idiomas*cant_secc*cant_sesiones            
    threshold_val   = -10             ## Valor mínimo de evaluación para aceptar la propuesta de horario
    
    last_best_eval  = 0 
    memoria_best_eval = []
    poblac_init,cant_bin_per_gen = crear_poblacion(cant_poblacion, cant_horas,cant_dias,cant_idiomas,cant_secc,cant_sesiones)

    ## Inician las iteraciones
    for i_iterac in range(max_iteraciones):
        actual_eval  = eval_poblac_fitness(poblac_init,cant_horas,cant_dias,cant_idiomas,cant_secc,cant_sesiones,cant_bin_per_gen)
        memoria_best_eval.append(actual_eval)
    print(poblac_init.shape)
    pass
# %%
