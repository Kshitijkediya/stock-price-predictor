from flask import Flask, render_template, redirect, url_for, request, flash, make_response
from models import db, User, Portfolio, Transaction
from stock_data import make_prediction, get_stock_data
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from datetime import datetime

app = Flask(
    __name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)
app.config['SECRET_KEY'] = '7ebd57c8178cdc606bdf75240d861c17' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    # Use db.session.get for primary key lookups as recommended
    return db.session.get(User, int(user_id))

# Context processor to inject theme
@app.context_processor
def inject_theme():
    theme = request.cookies.get('theme', 'dark-mode') # Default to dark mode
    return dict(theme=theme)

# Home route - Modified to handle explanations
@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        model_choice = request.form.get('model_choice')

        if not ticker:
            flash('Please enter a stock ticker.', 'warning')
            return render_template('index.html')

        # --- Call make_prediction, now returns predictions and explanations ---
        predictions, explanations = make_prediction(ticker, model_choice)

        if 'error' in predictions:
            flash(predictions['error'], 'danger') # Show error as flash message
            return render_template('index.html', ticker=ticker, model_choice=model_choice)

        # --- Pass explanations to the template ---
        return render_template(
            'index.html',
            predictions=predictions,
            explanations=explanations, # Pass the explanations dict
            ticker=ticker,
            model_choice=model_choice
        )
    # Initial GET request
    return render_template('index.html')

# --- Signup, Login, Logout routes remain unchanged ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('signup'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            # Redirect to the page user intended to visit, or home
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login unsuccessful. Please check username and password.', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# --- Preferences, Dashboard, Portfolio routes remain unchanged ---
# --- (Assuming get_stock_data is correctly used in portfolio calculations) ---
@app.route('/preferences', methods=['GET', 'POST'])
@login_required
def preferences():
    if request.method == 'POST':
        preferences_data = request.form.get('preferences')
        current_user.preferences = preferences_data
        db.session.commit()
        flash('Preferences saved!', 'success')
        # Redirect to GET to avoid form resubmission
        return redirect(url_for('preferences'))
    return render_template('preferences.html', preferences=current_user.preferences)

@app.route('/dashboard')
@login_required
def dashboard():
    # Eager load portfolios to avoid N+1 query issues if accessing portfolio names in template
    user_portfolios = current_user.portfolios # Assumes lazy='dynamic' or default lazy='select'
    return render_template('dashboard.html', username=current_user.username, portfolios=user_portfolios)

@app.route('/create_portfolio', methods=['GET', 'POST'])
@login_required
def create_portfolio():
    if request.method == 'POST':
        name = request.form['name']
        if not name:
            flash('Portfolio name cannot be empty.', 'warning')
        else:
            new_portfolio = Portfolio(name=name, user_id=current_user.id)
            db.session.add(new_portfolio)
            db.session.commit()
            flash('Portfolio created successfully!', 'success')
            return redirect(url_for('portfolio', portfolio_id=new_portfolio.id))
    return render_template('create_portfolio.html')


@app.route('/portfolio/<int:portfolio_id>', methods=['GET', 'POST'])
@login_required
def portfolio(portfolio_id):
    # Use get_or_404 for cleaner handling of missing portfolios
    portfolio = db.session.get(Portfolio, portfolio_id) # Use recommended .get
    if not portfolio:
        flash('Portfolio not found.', 'danger')
        return redirect(url_for('dashboard'))

    if portfolio.owner != current_user:
        flash('You do not have access to this portfolio.', 'danger')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        ticker = request.form.get('ticker')
        shares_str = request.form.get('shares')
        transaction_type = request.form.get('transaction_type')

        # Validate input
        if not all([ticker, shares_str, transaction_type]):
             flash('Missing form data for transaction.', 'warning')
             return redirect(url_for('portfolio', portfolio_id=portfolio.id))

        try:
            shares = int(shares_str)
            if shares <= 0:
                raise ValueError("Shares must be positive.")
        except ValueError:
            flash('Invalid number of shares.', 'danger')
            return redirect(url_for('portfolio', portfolio_id=portfolio.id))

        try:
            stock_data_df = get_stock_data(ticker)
            if stock_data_df.empty:
                flash(f'Could not retrieve data for ticker {ticker}.', 'danger')
                return redirect(url_for('portfolio', portfolio_id=portfolio.id))
            price = stock_data_df['Close'].iloc[-1]

            transaction = Transaction(
                portfolio_id=portfolio.id,
                ticker=ticker.upper(), # Standardize ticker
                shares=shares,
                price=price,
                transaction_type=transaction_type
            )

            total_value = price * shares
            if transaction_type == 'BUY':
                if portfolio.cash_balance < total_value:
                    flash('Insufficient cash balance.', 'danger')
                    # No need for rollback if commit hasn't happened
                    return redirect(url_for('portfolio', portfolio_id=portfolio.id))
                portfolio.cash_balance -= total_value
            else: # SELL
                # Check if user actually holds enough shares to sell
                current_holdings = calculate_holdings(portfolio.transactions)
                held_shares = next((h['shares'] for h in current_holdings if h['ticker'] == ticker.upper()), 0)
                if held_shares < shares:
                     flash(f'You only hold {held_shares} shares of {ticker.upper()}, cannot sell {shares}.', 'danger')
                     return redirect(url_for('portfolio', portfolio_id=portfolio.id))
                portfolio.cash_balance += total_value

            db.session.add(transaction)
            db.session.commit()
            flash('Transaction successful!', 'success')

        except ValueError as ve: # Catch specific errors from get_stock_data
             flash(f'Error processing transaction: {ve}', 'danger')
             db.session.rollback() # Rollback if commit hasn't happened but changes were staged
        except Exception as e:
            db.session.rollback()
            flash(f'An unexpected error occurred: {e}', 'danger')

        return redirect(url_for('portfolio', portfolio_id=portfolio.id)) # Redirect after POST

    # GET request: Fetch data for display
    transactions = portfolio.transactions # Use relationship directly
    holdings = calculate_holdings(transactions) # Pass the fetched transactions
    return render_template('portfolio.html', portfolio=portfolio, holdings=holdings, transactions=transactions)

# Helper function moved here for clarity, ensure get_stock_data is available
def calculate_holdings(transactions):
    holdings_dict = {}
    for txn in transactions:
        ticker = txn.ticker.upper()
        if ticker not in holdings_dict:
            holdings_dict[ticker] = 0
        if txn.transaction_type == 'BUY':
            holdings_dict[ticker] += txn.shares
        else: # SELL
            holdings_dict[ticker] -= txn.shares

    holdings_list = []
    for ticker, shares in holdings_dict.items():
        if shares <= 0: # Ignore tickers with zero or negative shares
            continue
        try:
            stock_data_df = get_stock_data(ticker) # Fetch fresh price
            current_price = stock_data_df['Close'].iloc[-1] if not stock_data_df.empty else 0
        except Exception as e:
            print(f"Error fetching price for {ticker} in calculate_holdings: {e}")
            current_price = 0 # Default price if fetch fails

        total_value = current_price * shares
        holdings_list.append({
            'ticker': ticker,
            'shares': shares,
            'current_price': round(current_price, 2),
            'total_value': round(total_value, 2)
        })
    return holdings_list


# Theme toggle route
@app.route('/toggle_theme')
def toggle_theme():
    current_theme = request.cookies.get('theme', 'dark-mode')
    new_theme = 'light-mode' if current_theme == 'dark-mode' else 'dark-mode'
    # Use request.referrer for safer redirection back to the previous page
    resp = make_response(redirect(request.referrer or url_for('home')))
    # Set cookie expiry for persistence (e.g., 30 days)
    max_age_seconds = 30 * 24 * 60 * 60
    resp.set_cookie('theme', new_theme, max_age=max_age_seconds, httponly=True, samesite='Lax')
    return resp


if __name__ == '__main__':
    # Consider disabling debug mode for production
    app.run(debug=True) # Keep debug=True for development