#%%
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import matplotlib as plt
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
"""
iDIOmAS = [ing , spa , .... ARB]   10
IDIONA XYZ= {  A ,  B}              2
SECCION_XYZ= {19-21, 21-23}         2 
CROMOSOMA = GEN1 | GEN2 ....  |GEN40 = bits >> 160

GEN = [1,2,3,4] IDOMA XYZ A1 A2 B1 B2
DIAS [ l - s]         6
horas [19-21, 21-23]  2   

12cajas = [0,1,2,3,...11] _ _ _ _
"""
# %%
def crear_poblacion(cant_poblacion, cant_horas,cant_dias,cant_idiomas,cant_secc,cant_sesiones):
    cant_genes       = cant_idiomas*cant_secc*cant_sesiones  
    cant_bin_per_gen = 0
    for num in [cant_horas*cant_dias]:
        cant_bin_per_gen += len(str(bin(num))[2:]) 
    temp_pobl = np.random.random_integers(low= 0, high=1,size=(cant_poblacion,cant_bin_per_gen*cant_genes) )
    return temp_pobl,cant_bin_per_gen

def decodificador(poblacion,cant_horas,cant_dias,cant_idiomas,cant_secc,cant_sesiones,cant_bin_per_gen):
    def sess_bin_dec(num,cant_sessiones_unicas):
        """
        num es un vector = [1,0,1,0,...,1] que representa un número binario
        cant_sessiones_unicas = valor que consiste en multiplicar HORARIOS DISPONIBLES POR DÍA * DÍAS QUE SE PUEDEN DICTAR
        """
        mult = np.stack([2**(i) for i in range(len(num))][::-1])
        dec = np.dot(num,mult)
        while dec>=cant_sessiones_unicas:
            dec = dec%cant_sessiones_unicas if dec>=cant_sessiones_unicas else dec
        return dec 
    cant_genes      = cant_idiomas*cant_secc*cant_sesiones   
    cant_sessiones_unicas = cant_horas*cant_dias
    num_individuos = poblacion.shape[0]
    poblacion_deco = np.zeros((num_individuos,cant_genes))
    for ind, individuo in enumerate(poblacion):
        poblacion_deco[ind] = [sess_bin_dec(individuo[ini_bin:(ini_bin+cant_bin_per_gen)],cant_sessiones_unicas) for ini_bin in range(cant_genes)]
    return poblacion_deco[0]
    
def eval_poblac_fitness(poblacion,cant_horas,cant_dias,cant_idiomas,cant_secc,cant_sesiones,cant_bin_per_gen):
    """
    Se han reconocido 3 tipos de cruces de horarios
    1) Cruces entre secciones - mismo curso        = x1000
    2) Cruces entre secciones - diferenctes curso  = x1
    3) Cruces entre sesiones del mismo curso       = x100
    Función de evaluación:
    f(x1 + x2 + x3) = -(x1*1000+x2+x3*100)
    ## EVALUACIÓN
    1 - (100)comprobar traslape entre seSSiones del mismo curso    =>> verificar que los casilleros de las seSSiones sean diferentes | Dentro de la misma sección A o B de cada curso, comprobar que las dos sesiones dentro de la sección A o B sean diferentes.
        EJEMP: La sesión1 de la semana se traslapan con la sesión2 de la semana
    2 - (100)comprobar traslape entre seCCiones del mismo curso    =>> verificar que los casilleros de las seSSiones sean diferentes | Agrupando cada seCCión A y B para un mismo curso, comprobar similitud de casillero de sesión entre A y B del mismo curso
        EJEMP: La sección A y B del español se traslapan
    3 - (1)   comprobar traslape entre seCCiones de DIFERENTE curso =>> verifivar que los casilleros de las seSSiones sean diferentes | Agrupando cada curso con su sección A y B, para compararla con la de otro
        EJEMP: La sección AóB del español se traslapa con la sección AóB del Ingles en alguna sesión de la semana
    """
    # Función auxiliar para trasnformar un gen en un número de casillero de sesión en el calendario
    def sess_bin_dec(num,cant_sessiones_unicas):
        """
        num es un vector = [1,0,1,0,...,1] que representa un número binario
        cant_sessiones_unicas = valor que consiste en multiplicar HORARIOS DISPONIBLES POR DÍA * DÍAS QUE SE PUEDEN DICTAR
        """
        mult = np.stack([2**(i) for i in range(len(num))][::-1])
        dec = np.dot(num,mult)
        while dec>=cant_sessiones_unicas:
            dec = dec%cant_sessiones_unicas if dec>=cant_sessiones_unicas else dec
        return dec 
    # Evalua el criterio 1
    # def determinar_igualdad_session(hr1,hr2):
    #     return hr1==hr2
    # vf_horario = np.vectorize(determinar_igualdad_session)

    # Evalua el criterio 2 incluye a la evaluación 1
    def determinar_igualdad_session_secc(hr1,hr2,hr3,hr4):
        temp = [hr1,hr2,hr3,hr4]
        igualdad = [hr_p == hr_q for ind, hr_p in enumerate(temp[:-1]) for hr_q in temp[ind+1:]]
        return sum(igualdad)
    vf_horario_secc = np.vectorize(determinar_igualdad_session_secc)

    # Evalua el criterio 3

    # DECODIFICANDO CROMOSOMA
    cant_genes      = cant_idiomas*cant_secc*cant_sesiones   
    cant_sessiones_unicas = cant_horas*cant_dias
    num_individuos = poblacion.shape[0]
    num_num_bin    = poblacion.shape[1]
    poblacion_deco = np.zeros((num_individuos,cant_genes))
    for ind, individuo in enumerate(poblacion):
        poblacion_deco[ind] = [sess_bin_dec(individuo[ini_bin:(ini_bin+cant_bin_per_gen)],cant_sessiones_unicas) for ini_bin in range(cant_genes)]
    #print(poblacion_deco)
    ## Evaluando el primer tipo de cruce | traslape entre sesiones de una misma sección
    # agrupado_sess = np.reshape(poblacion_deco, (cant_idiomas,-1,cant_sesiones))
    # eval_1 = np.sum(vf_horario(agrupado_sess[:,:,0],agrupado_sess[:,:,1]), axis=1)

    ## Evaluando el segundo tipo de cruce | traslape entre secciones de un mismo curso
    agrupado_idioma_sess = np.reshape(poblacion_deco, (num_individuos,-1,cant_sesiones*cant_secc))
    eval_2 = np.sum(vf_horario_secc(agrupado_idioma_sess[:,:,0],agrupado_idioma_sess[:,:,1],agrupado_idioma_sess[:,:,2],agrupado_idioma_sess[:,:,3]), axis=1)
    
    ## Evaluando el tercer tipo de cruce | traslape entre secciones de diferentes cursos
    eval_3 = []
    for ind, individuo_x in enumerate(agrupado_idioma_sess):
        eval_ = 0
        for i, idioma_i in enumerate(individuo_x[:-1]):
            for idioma_j in individuo_x[i+1:]:
                for sess_i in idioma_i:
                    eval_+=sum(sess_i == idioma_j)
                    #print(eval_)
        eval_3.append(eval_)
    eval_3 = np.stack(eval_3)

    ## Computando la evaluación general de cada individuo
    eval_total = -(eval_2*100 + eval_3)
    

    #print(vf_horario(agrupado_sess[:,:,0],agrupado_sess[:,:,1]))
    #return np.sum(vf_horario(agrupado_sess[:,:,0],agrupado_sess[:,:,1]), axis=1)
    return eval_total , np.max(eval_total)

def seleccion(poblac_init,fitness_eval_result,threshold_padres):
    #print(fitness_eval_result)
    padres = poblac_init.copy()
    fitness_eval = fitness_eval_result.copy()
    num_individuos = poblac_init.shape[0]
    num_num_bin    = poblac_init.shape[1]
    #Se reemplaza el peor cromosoma por el mejor
    poblac_init[np.argmin(fitness_eval_result)]=poblac_init[np.argmax(fitness_eval_result)]
    fitness_eval[np.argmin(fitness_eval_result)]=fitness_eval[np.argmax(fitness_eval_result)]
    #Se ordenan los resultados (mayor=>menor)
    fitness_sort = np.sort(fitness_eval)[::-1]
    #Se escoge a los X primeros
    n_primeros = round(num_individuos*(100-threshold_padres)/100)
    
    best_n = fitness_sort[:n_primeros]
    #print(best_n)
    for ind in range(num_individuos):
        if not fitness_eval[ind] in best_n:
            ind_selected = np.where(fitness_eval == random.choice(best_n))[0][0] 
            #print(ind_selected)
            padres[ind] = poblac_init[ind_selected ]
            fitness_eval[ind] = fitness_eval[ind_selected]

    return padres

def crossover(padres,cant_bin_per_gen,cant_genes):
    num_individuos = padres.shape[0]
    num_num_bin    = padres.shape[1]
    e=0
    puntos_crossover=np.array([cant_bin_per_gen*2*i for i in range(cant_genes//2)])[1:]
    #print(puntos_crossover)
    auxiliar=np.zeros((num_individuos,num_num_bin))
    for i in range(0,num_individuos,2):
        #print(i)
        hijo1=np.append(padres[i  ][:puntos_crossover[e]] ,  padres[i+1][puntos_crossover[e]:])
        hijo2=np.append(padres[i+1][:puntos_crossover[e]] ,  padres[i  ][puntos_crossover[e]:])
        auxiliar[i]=hijo1
        auxiliar[i+1]=hijo2
        e=e+1
    return auxiliar

def mutacion(crossover,factor_mutacion):
    mutado = crossover.copy()
    num_individuos = crossover.shape[0]
    num_num_bin    = crossover.shape[1]
    for _ in range(factor_mutacion):
        for i in range(num_individuos):
            pos_aleatoria=random.randrange(0,num_num_bin)
            mutado[i][pos_aleatoria] = 1- mutado[i][pos_aleatoria]
    return mutado

def stop_criterio(poblacion, fitness_eval_result,threshold_val,memoria_best_eval, threshold_std_eval):
    curr_max_eval = np.max(fitness_eval_result)
    memoria = memoria_best_eval.copy()
    memoria.sort()
    if len(memoria)>1000:
        memoria = np.stack(memoria[-10:])
        memoria_std = np.std(memoria)
        if memoria_std <= threshold_std_eval:
            print("Criterio de la desviación Estandar")
            return True
    if threshold_val <= curr_max_eval:
        print("Criterio de valor óptimo menor a {}".format(threshold_val))
        return True

def print_eval_result(memoria_best_eval):
    temp_df = pd.DataFrame(memoria_best_eval)
    fig, ax = plt.subplots(figsize=(15,10))
    ff =temp_df.plot(kind='line',ax=ax)
    ff.set_title('Best Fitness by Iteration', size= 28 ) 
    ff.set_xlabel('Iteration',size= 20 )
    ff.set_ylabel('Fitness Evaluation',size= 20 )

def print_producto(poblacion, fitness_eval_result,cant_horas,cant_dias,cant_idiomas,cant_secc,cant_sesiones,cant_bin_per_gen,HORARIOS):
    tabla = pd.DataFrame(np.zeros((cant_horas,cant_dias)))
    tabla.columns = ['LUNES', 'MARTES', 'MIERCOLES', 'JUEVES', 'VIERNES', 'SABADO']
    IDIOMAS   = ['ING', 'ALM', 'FRAN', 'ITA', 'PORT', 'CH','RS', 'JAP', 'ARAB', 'ESP' ]

    tabla = tabla.replace(0,'')
    #print(tabla)
    curr_max_eval = np.max(fitness_eval_result)
    best_individuo = poblacion[np.where(fitness_eval_result == curr_max_eval)]

    best_individuo = decodificador(best_individuo,cant_horas,cant_dias,cant_idiomas,cant_secc,cant_sesiones,cant_bin_per_gen)
    best_individuo_ind = np.reshape(best_individuo, (cant_idiomas,-1))

    #best_individuo = np.stack([HORARIOS[int(i)] for i in best_individuo])
    #best_individuo = np.reshape(best_individuo, (cant_idiomas,-1))
    prefix = ''
    for idioma, ind in enumerate(best_individuo_ind):
        #print(ind)
        #print('===================')
        prefix = IDIOMAS[idioma]
        for secc, pos in enumerate(ind):
            if secc == 0 or secc == 1:
                prefix_secc = prefix +'-A'
            else:
                prefix_secc = prefix + '-B'
            if pos< cant_dias:
                #print(pos)
                tabla.iloc[0,int(pos)] = tabla.iloc[0,int(pos)] + prefix_secc + HORARIOS[0]+'\n'
            else:
                #print(int(pos-cant_dias))
                tabla.iloc[1,int(pos-cant_dias)] = tabla.iloc[1,int(pos-cant_dias)] + prefix_secc + HORARIOS[1]+'\n'
    tabla.to_csv(path_or_buf= 'best.csv', index=False    )
    print("Mejor Individuo: ")
    print(tabla)
    print("Mejor Puntaje")
    print(curr_max_eval)

#%%
if __name__ == "__main__":
    max_iteraciones = 1000  ## Cantidad máxima de iteraciones si no llega a cumplir ningun criterio
    cant_poblacion  = 10    ## Cantidad de individuos por población
    cant_horas      = 2     #['19-21','21-23' ] ## Cada 2 horas sin traslape de horas
    cant_dias       = 6     #['L','Mt','Mr','J','V','S']## Días disponibles para dar clases a la semana
    cant_idiomas    = 10    ## Cantidad de idiomas
    cant_secc       = 2     ## canridad de secciones por curso
    cant_sesiones   = 2     ## Cantidad de sesiones por semana
    cant_genes      = cant_idiomas*cant_secc*cant_sesiones            
    threshold_val   = -50  ## Valor mínimo de evaluación para aceptar la propuesta de horario
    threshold_std_eval = 3 ## Desviación standar entre las 10 ultimas iteraciones para parar el ajuste
    threshold_padres= 30   ## Porcentaje de la población que se aniquila para la formación de padres
    factor_mutacion = 2    ## Cuantos procesos de mutación de bits por individuo mutaran
    HORARIOS = ['|19-21','|21-23']
    
    last_best_eval  = 0 
    memoria_best_eval = []
    poblac_init,cant_bin_per_gen = crear_poblacion(cant_poblacion, cant_horas,cant_dias,cant_idiomas,cant_secc,cant_sesiones)
    #print(poblac_init.shape)
    ## Inician las iteraciones

    for i_iterac in tqdm(range(max_iteraciones)):
        fitness_eval_result, actual_best  = eval_poblac_fitness(poblac_init,cant_horas,cant_dias,cant_idiomas,cant_secc,cant_sesiones,cant_bin_per_gen)
        memoria_best_eval.append(actual_best)
        if stop_criterio(poblac_init, fitness_eval_result,threshold_val,memoria_best_eval, threshold_std_eval) or i_iterac == max_iteraciones-1:
            print_producto(poblac_init, fitness_eval_result,cant_horas,cant_dias,cant_idiomas,cant_secc,cant_sesiones,cant_bin_per_gen,HORARIOS)
            print_eval_result(memoria_best_eval)
            break
        padres_i    = seleccion(poblac_init,fitness_eval_result,threshold_padres)
        crossover_i = crossover(padres_i,cant_bin_per_gen,cant_genes)
        mutacion_i  = mutacion(crossover_i,factor_mutacion)
        poblac_init = mutacion_i.copy()
        last_best_eval = actual_best
    print(memoria_best_eval[-30:])
    pass
# %%