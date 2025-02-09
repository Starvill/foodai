import requests

# Ваш API-ключ
API_KEY = "YOUR_API_KEY"

# Функция поиска продуктов
def search_food(query):
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": query,
        "pageSize": 1,
        "api_key": API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['foods']:
            return data['foods'][0]  # Первый результат
    return None

# Функция получения информации о калориях
def get_calories(fdc_id):
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
    params = {"api_key": API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        for nutrient in data.get("foodNutrients", []):
            if nutrient.get("nutrientName") == "Energy" and nutrient.get("unitName") == "kcal":
                return nutrient.get("value")
    return None

# Пример использования
if __name__ == "__main__":
    food_query = "pizza"
    food_data = search_food(food_query)
    if food_data:
        print(f"Найден продукт: {food_data['description']}")
        fdc_id = food_data['fdcId']
        calories = get_calories(fdc_id)
        if calories is not None:
            print(f"Калорийность: {calories} ккал на 100 г")
        else:
            print("Калорийность не найдена.")
    else:
        print("Продукт не найден.")