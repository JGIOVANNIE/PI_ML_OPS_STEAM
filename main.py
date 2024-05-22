# relizaremos el deploy de nuestras consultas 
from fastapi import FastAPI
from typing import List, Dict, Union, Any
import pandas as pd
import pyarrow
import numpy as np 


app = FastAPI(title='STEAM Games', description='Esta es una aplicación para realizar consultas sobre todo el mundo de STEAM.')

df_UFG = pd.read_parquet('./API/UserForGenre.parquet')
df_UsersR = pd.read_parquet('./API/UsersR.parquet')
modelo = pd.read_parquet('./API/modelo.parquet')


@app.get("/play_time_genre/{genero}")
async def play_time_genre(genero: str) -> Dict[str, int]:
    # Filtrar los datos por género
        df_genero = df_UFG[df_UFG['genres'].str.lower().str.contains(genero.lower())]

        if df_genero.empty:
            return {"mensaje": f"No hay datos para el género: {genero}"}

    # Agrupar los datos por año y sumar las horas jugadas
        df_agrupado = df_genero.groupby('año')['playtime_forever'].sum().reset_index()

    # Encontrar el año con más horas jugadas
        max_playtime_year = df_agrupado.loc[df_agrupado['playtime_forever'].idxmax(), 'año']

        return {
        f"Año de lanzamiento con más horas jugadas para Género {genero}": max_playtime_year
        }

@app.get("/user_for_genre/{genero}")
async def user_for_genre(genero: str) -> Dict[str, Union[str, List[Dict[str, float]]]]:
        # Filtrar los datos por género
        df_genero = df_UFG[df_UFG['genres'].str.contains(genero.lower(), na=False)]

        if df_genero.empty:
            return {"mensaje": f"No hay datos para el género: {genero}"}

        # Agrupar los datos por usuario y año
        df_agrupado = df_genero.groupby(['user_id', 'año']).agg({'playtime_forever': 'sum'}).reset_index()

        # Encontrar el usuario con más horas jugadas
        max_playtime_user = df_agrupado.loc[df_agrupado['playtime_forever'].idxmax(), 'user_id']

        # Crear una lista de la acumulación de horas jugadas por año para ese usuario
        user_data = df_agrupado[df_agrupado['user_id'] == max_playtime_user]
        user_data = user_data.rename(columns={'playtime_forever': 'Horas'})
        horas_jugadas = user_data[['año', 'Horas']].to_dict('records')

        return {
            "Usuario con más horas jugadas para el género proporcionado es": max_playtime_user,
            "Horas jugadas": horas_jugadas
        }

@app.get('/users_recommend/{anio}')
async def users_recommend(anio: int) -> List[Dict[str, str]]:
        # Filtrar las reseñas que son recomendaciones y tienen un análisis de sentimiento positivo o neutral
        filtro = df_UsersR[(df_UsersR['recommend'] == True) & (df_UsersR['sentiment_analysis'].isin([1, 2]))]

        # Filtrar los juegos que se lanzaron en el año dado
        filtro = filtro[filtro['anio'] == anio]

        # Contar las recomendaciones para cada juego
        cuenta_de_recomendaciones = filtro['title'].value_counts()

        # Obtener los tres juegos con más recomendaciones, si hay al menos tres juegos
        if len(cuenta_de_recomendaciones) >= 3:
            top_juegos = cuenta_de_recomendaciones.nlargest(3).index.tolist()
            return [
                {"Puesto 1": top_juegos[0]},
                {"Puesto 2": top_juegos[1]},
                {"Puesto 3": top_juegos[2]}
            ]
        else:
            return [{"error": "No hay suficientes juegos con recomendaciones para el año dado."}]


@app.get('/users_not_recommend/{anio}')
async def users_not_recommend(anio: int) -> List[Dict[str, str]]:
        # Filtrar las reseñas que no son recomendaciones y tienen un análisis de sentimiento negativo
        filtrado = df_UsersR[(df_UsersR['recommend'] == False) & (df_UsersR['sentiment_analysis'].isin([0]))]

        # Filtrar los juegos que se lanzaron en el año dado
        filtrado = filtrado[filtrado['anio'] == anio]

        # Contar las no recomendaciones para cada juego
        cuenta_negativa = filtrado['title'].value_counts()

        # Devolver los tres juegos con más no recomendaciones
        
        if len(cuenta_negativa) >= 3:
            top_juegos_norecomendados = cuenta_negativa.nlargest(3).index.tolist()
            return [
                {"Puesto 1": top_juegos_norecomendados[0]},
                {"Puesto 2": top_juegos_norecomendados[1]},
                {"Puesto 3": top_juegos_norecomendados[2]}
            ]
        else:
            return [{"error": "No hay suficientes juegos no recomendados para el año dado."}]

@app.get('/sentiment_analysis/{anio}')
async def sentiment_analysis(anio: int) -> Dict[str, Dict[str, int]]:
        # Filtramos el dataframe por año
        df_filtrado = df_UsersR[df_UsersR['anio'] == anio]

        # Contar la cantidad de registros de reseñas que se encuentren categorizados con un análisis de sentimiento
        contar_sentimientos = df_filtrado['sentiment_analysis'].value_counts()

        # Crear un nuevo diccionario con el formato deseado
        sentimientos = {
            f"En el año {anio} hubo comentarios": {
                "positivos": contar_sentimientos.get(2, 0),
                "neutrales": contar_sentimientos.get(1, 0),
                "negativos": contar_sentimientos.get(0, 0)
            }
        }

        return sentimientos
 
@app.get("/recomendacion usuario/{id_usuario}")
async def recomendacion_usuario(id_usuario: str) -> dict:
    # Obtener los juegos recomendados para el usuario
    juegos_recomendados = modelo.columns[np.where(modelo.loc[id_usuario] == 0)][:5]
    # Crear un diccionario con el formato deseado
    resultado_recomendacion = {
        "Las recomendaciones el usuario son:": list(juegos_recomendados)
    }

    return resultado_recomendacion        
  
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
